import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from itertools import product
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader

from training_model.diabetes_model import DiabetesModel, VALID_MODEL_TYPES
from training_model.glucose_dataset import GlucoseDataset
from training_model.preparing.static_preprocessing import StaticProcessing
from training_model.repository import Repository
from utils.make_windows import MakeWindows

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse(pred, target, mask):
    """MSE loss only where the drug is actually prescribed (mask == 1)."""
    active = mask.sum()
    if active == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return ((pred - target) ** 2 * mask).sum() / active


def compute_measurement_stats(windows):
    """Compute per-channel mean/std of CGM, CBG, blood_ketone from training windows."""
    all_values = [[], [], []]
    for w in windows:
        all_values[0].extend(v for v in w['cgm_values'] if v != 0.0)
        all_values[1].extend(v for v in w['cbg_values'] if v != 0.0)
        all_values[2].extend(v for v in w['blood_ketones'] if v != 0.0)

    means, stds = [], []
    for ch in all_values:
        arr = np.array(ch) if ch else np.array([0.0])
        means.append(float(arr.mean()))
        stds.append(float(max(arr.std(), 1e-8)))
    return means, stds


def compute_max_doses(windows):
    """Compute max insulin / tablet dose from training windows only."""
    ins = [d for w in windows for d in w['insulin_doses_by_type'].values()]
    tab = [d for w in windows for d in w['drug_tablets_by_type'].values()]
    return (max(ins) if ins else 1.0), (max(tab) if tab else 1.0)


def split_by_patient(windows, train_ratio=0.8, val_ratio=0.1):
    """Patient-level split preserving temporal integrity."""
    pids = sorted(set(w['patient_id'] for w in windows))
    random.shuffle(pids)

    n_train = int(len(pids) * train_ratio)
    n_val = int(len(pids) * val_ratio)

    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train:n_train + n_val])

    train_w = [w for w in windows if w['patient_id'] in train_pids]
    val_w = [w for w in windows if w['patient_id'] in val_pids]
    test_w = [w for w in windows if w['patient_id'] not in train_pids and w['patient_id'] not in val_pids]

    return train_w, val_w, test_w, train_pids


# ══════════════════════════════════════════════════════════════════════
#  Helpers for forwarding a batch through the model
# ══════════════════════════════════════════════════════════════════════

def _forward_batch(model, batch, device):
    """Unpack a collated batch and run a forward pass. Returns 4 output tensors."""
    xgb = batch.get('xgb_preds')
    if xgb is not None:
        xgb = xgb.to(device)

    return model(
        batch['measurements'].to(device),
        batch['static'].to(device),
        batch['drug_idx'].to(device),
        batch['comorb_idx'].to(device),
        batch['food_intake'].to(device),
        batch['therapy_type'].to(device),
        xgb_preds=xgb,
    )


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

class FullModel:
    def __init__(self):
        self.repo = Repository()
        self._data_cache = None

    def _load_data_once(self):
        """Query DB once and cache all raw data needed for training."""
        if self._data_cache is not None:
            return self._data_cache

        patients = self.repo.get_patients_td()
        patient_to_drugs = self.repo.get_patient_drugs_map()
        patient_to_comorbities = self.repo.get_patient_comorbities_map()
        unique_drugs = StaticProcessing.get_unique_entities(list(patient_to_drugs.values()))
        unique_comorbities = StaticProcessing.get_unique_entities(list(patient_to_comorbities.values()))

        patient_ids, raw_features, drug_indices, comorb_indices = (
            StaticProcessing.extract_patient_data(
                patients, unique_drugs, unique_comorbities,
                patient_to_drugs, patient_to_comorbities
            )
        )

        insulin_recs = self.repo.get_taking_insulin()
        tablet_recs = self.repo.get_taking_tablets()
        meas_recs = self.repo.get_measurements()
        food_recs = self.repo.get_dietary()
        windows = MakeWindows().build_feature_windows(insulin_recs, tablet_recs, meas_recs, food_recs)
        windows.sort(key=lambda w: (w['patient_id'], w['window_start']))

        self._data_cache = {
            'patient_ids': patient_ids,
            'raw_features': raw_features,
            'drug_indices': drug_indices,
            'comorb_indices': comorb_indices,
            'unique_drugs': unique_drugs,
            'unique_comorbities': unique_comorbities,
            'windows': windows,
            'num_insulin': len(self.repo.get_insulin_list()),
            'num_drug_types': len(self.repo.get_tablets_list()),
        }
        return self._data_cache

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_and_eval(self, hidden_size, dropout, lr, batch_size,
                       emb_dim=32, num_rnn_layers=1, epochs=30, patience=5,
                       model_type="gru"):
        set_seed()
        data = self._load_data_once()

        train_w, val_w, test_w, train_pids = split_by_patient(data['windows'])

        # Fit scaler on TRAINING patients only
        train_feat_idx = [i for i, pid in enumerate(data['patient_ids']) if pid in train_pids]
        scaler = StaticProcessing.fit_scaler([data['raw_features'][i] for i in train_feat_idx])

        static_dict = StaticProcessing.build_static_dict(
            data['patient_ids'], data['raw_features'],
            data['drug_indices'], data['comorb_indices'], scaler
        )

        meas_mean, meas_std = compute_measurement_stats(train_w)
        max_insulin, max_tablet = compute_max_doses(train_w)

        num_ins = data['num_insulin']
        num_tabs = data['num_drug_types']

        # ── XGBoost pre-training for hybrid mode ────────────────────
        xgb_feature_dim = 0
        if model_type == "hybrid_xgb_lstm":
            from training_model.xgboost_baseline import XGBoostBaseline

            xgb_baseline = XGBoostBaseline(num_ins, num_tabs)
            xgb_baseline.fit(train_w, static_dict, max_insulin, max_tablet)
            xgb_preds = xgb_baseline.predict_all(static_dict)
            xgb_feature_dim = xgb_baseline.output_dim

            for pid, pred_vec in xgb_preds.items():
                static_dict[pid]['xgb_preds'] = torch.tensor(pred_vec, dtype=torch.float32)

        # ── build datasets ──────────────────────────────────────────
        ds_kwargs = dict(
            static_dict=static_dict,
            num_insulin=num_ins,
            num_diabetes_tablets=num_tabs,
            max_insulin=max_insulin,
            max_tablet=max_tablet,
            meas_mean=meas_mean,
            meas_std=meas_std,
        )
        train_ds = GlucoseDataset(train_w, **ds_kwargs)
        val_ds = GlucoseDataset(val_w, **ds_kwargs)
        test_ds = GlucoseDataset(test_w, **ds_kwargs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=GlucoseDataset.collate_fn)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=GlucoseDataset.collate_fn)

        static_dim = len(data['raw_features'][0])

        model = DiabetesModel(
            static_dim=static_dim,
            hidden_size=hidden_size,
            num_insulin_types=num_ins,
            num_drug_types=num_tabs,
            unique_drugs_size=len(data['unique_drugs']),
            unique_comorbities_size=len(data['unique_comorbities']),
            emb_dim=emb_dim,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
            model_type=model_type,
            xgb_feature_dim=xgb_feature_dim,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        bce_fn = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        wait = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()

                pred_ins, pred_tab, pred_ins_m, pred_tab_m = _forward_batch(model, batch, device)

                y_ins = batch['y_insulin'].to(device)
                y_tab = batch['y_diabetes_tablet'].to(device)
                y_ins_mask = batch['y_insulin_mask'].to(device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(device)

                loss = (
                    masked_mse(pred_ins, y_ins, y_ins_mask)
                    + masked_mse(pred_tab, y_tab, y_tab_mask)
                    + bce_fn(pred_ins_m, y_ins_mask)
                    + bce_fn(pred_tab_m, y_tab_mask)
                )
                loss.backward()
                optimizer.step()

            val_loss = self._eval_loss(model, val_loader, bce_fn, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        model.to(device)

        metrics = self._eval_metrics(model, val_loader, device)
        return metrics, model, test_ds

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_loss(model, loader, bce_fn, device):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                p_ins, p_tab, p_im, p_tm = _forward_batch(model, b, device)

                y_ins = b['y_insulin'].to(device)
                y_tab = b['y_diabetes_tablet'].to(device)
                y_im = b['y_insulin_mask'].to(device)
                y_tm = b['y_diabetes_tablet_mask'].to(device)

                loss = (
                    masked_mse(p_ins, y_ins, y_im)
                    + masked_mse(p_tab, y_tab, y_tm)
                    + bce_fn(p_im, y_im)
                    + bce_fn(p_tm, y_tm)
                )
                total += loss.item()
                n += 1
        return total / max(n, 1)

    @staticmethod
    def _eval_metrics(model, loader, device):
        model.eval()
        ys_ins, ps_ins, ms_ins = [], [], []
        ys_tab, ps_tab, ms_tab = [], [], []

        with torch.no_grad():
            for b in loader:
                p_ins, p_tab, _, _ = _forward_batch(model, b, device)

                ys_ins.append(b['y_insulin'])
                ps_ins.append(p_ins.cpu())
                ms_ins.append(b['y_insulin_mask'])
                ys_tab.append(b['y_diabetes_tablet'])
                ps_tab.append(p_tab.cpu())
                ms_tab.append(b['y_diabetes_tablet_mask'])

        y_i = torch.cat(ys_ins).numpy().flatten()
        p_i = torch.cat(ps_ins).numpy().flatten()
        m_i = torch.cat(ms_ins).numpy().flatten() > 0
        y_t = torch.cat(ys_tab).numpy().flatten()
        p_t = torch.cat(ps_tab).numpy().flatten()
        m_t = torch.cat(ms_tab).numpy().flatten() > 0

        r2_ins = r2_score(y_i[m_i], p_i[m_i]) if m_i.any() else 0.0
        r2_tab = r2_score(y_t[m_t], p_t[m_t]) if m_t.any() else 0.0
        mae_ins = mean_absolute_error(y_i[m_i], p_i[m_i]) if m_i.any() else 0.0
        mae_tab = mean_absolute_error(y_t[m_t], p_t[m_t]) if m_t.any() else 0.0

        val_loss = (
            ((y_i - p_i) ** 2 * m_i).sum() / max(m_i.sum(), 1)
            + ((y_t - p_t) ** 2 * m_t).sum() / max(m_t.sum(), 1)
        )
        return {
            'val_loss': float(val_loss),
            'r2_insulin': float(r2_ins),
            'r2_tablets': float(r2_tab),
            'mae_insulin': float(mae_ins),
            'mae_tablets': float(mae_tab),
        }

    # ------------------------------------------------------------------
    # Search entry-points
    # ------------------------------------------------------------------

    def main(self, search_method="optuna", n_trials=30, model_type="gru"):
        """
        :param search_method: "grid" or "optuna"
        :param n_trials: number of Optuna trials (ignored for grid)
        :param model_type: "gru", "lstm", or "hybrid_xgb_lstm"
        """
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}")

        self._load_data_once()
        print(f"Model type: {model_type}")

        best_model = None
        best_test_ds = None

        if search_method == "grid":
            grid = {
                "hidden_size": [64, 128],
                "dropout": [0.1, 0.2, 0.3],
                "lr": [1e-3, 3e-4],
                "batch_size": [16, 32],
            }
            best_loss = float("inf")
            best_config = None

            for hs, do, lr, bs in product(*grid.values()):
                metrics, model, test_ds = self.train_and_eval(
                    hs, do, lr, bs, model_type=model_type)
                print(
                    f"hidden={hs}, dropout={do}, lr={lr}, bs={bs} | "
                    f"loss={metrics['val_loss']:.4f}  "
                    f"R² ins={metrics['r2_insulin']:.4f}  R² tab={metrics['r2_tablets']:.4f}  "
                    f"MAE ins={metrics['mae_insulin']:.4f}  MAE tab={metrics['mae_tablets']:.4f}"
                )
                if metrics['val_loss'] < best_loss:
                    best_loss = metrics['val_loss']
                    best_config = dict(hidden_size=hs, dropout=do, lr=lr, batch_size=bs)
                    best_model = model
                    best_test_ds = test_ds

            print("Best Grid Config:", best_config)

        elif search_method == "optuna":
            holder = {'model': None, 'test_ds': None, 'loss': float('inf')}

            def objective(trial):
                hs = trial.suggest_int("hidden_size", 32, 256, log=True)
                nl = trial.suggest_int("num_rnn_layers", 1, 3)
                do = trial.suggest_float("dropout", 0.05, 0.5)
                lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                bs = trial.suggest_categorical("batch_size", [16, 32, 64])
                ed = trial.suggest_categorical("emb_dim", [16, 32, 64])

                metrics, model, test_ds = self.train_and_eval(
                    hs, do, lr, bs,
                    emb_dim=ed, num_rnn_layers=nl,
                    model_type=model_type,
                )
                if metrics['val_loss'] < holder['loss']:
                    holder.update(model=model, test_ds=test_ds, loss=metrics['val_loss'])
                return metrics['val_loss']

            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(objective, n_trials=n_trials)
            print("Best Optuna Config:", study.best_params)
            print(f"Best Val Loss: {study.best_value:.4f}")
            best_model = holder['model']
            best_test_ds = holder['test_ds']

        # ── Final test-set evaluation ───────────────────────────────
        if best_model is not None and best_test_ds is not None and len(best_test_ds) > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_loader = DataLoader(
                best_test_ds, batch_size=32, shuffle=False,
                collate_fn=GlucoseDataset.collate_fn)
            test_metrics = self._eval_metrics(best_model, test_loader, device)
            print("\n=== TEST SET RESULTS ===")
            print(f"Loss     : {test_metrics['val_loss']:.4f}")
            print(f"R² Ins   : {test_metrics['r2_insulin']:.4f}")
            print(f"R² Tab   : {test_metrics['r2_tablets']:.4f}")
            print(f"MAE Ins  : {test_metrics['mae_insulin']:.4f}")
            print(f"MAE Tab  : {test_metrics['mae_tablets']:.4f}")

            torch.save(best_model.state_dict(), "best_model.pt")
            print("Model saved to best_model.pt")


if __name__ == '__main__':
    pipeline = FullModel()
    # Switch model_type here: "gru", "lstm", or "hybrid_xgb_lstm"
    pipeline.main(search_method="optuna", n_trials=30, model_type="gru")
