import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from itertools import product
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score, accuracy_score,
)
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
        self._last_train_artifacts = {}

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
                       model_type="gru", loss_weight_bce=1.0):
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
        xgb_baseline = None
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

        pw_ins, pw_tab = self._compute_pos_weights(train_w, num_ins, num_tabs)
        bce_fn_ins = nn.BCEWithLogitsLoss(pos_weight=pw_ins.to(device))
        bce_fn_tab = nn.BCEWithLogitsLoss(pos_weight=pw_tab.to(device))

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

                mse_part = masked_mse(pred_ins, y_ins, y_ins_mask) + masked_mse(pred_tab, y_tab, y_tab_mask)
                bce_part = bce_fn_ins(pred_ins_m, y_ins_mask) + bce_fn_tab(pred_tab_m, y_tab_mask)
                loss = mse_part + loss_weight_bce * bce_part
                loss.backward()
                optimizer.step()

            val_loss = self._eval_loss(model, val_loader, bce_fn_ins, bce_fn_tab, device, loss_weight_bce)
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

        self._last_train_artifacts = {
            'model_config': dict(
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
            ),
            'scaler': scaler,
            'meas_mean': meas_mean,
            'meas_std': meas_std,
            'max_insulin': max_insulin,
            'max_tablet': max_tablet,
            'xgb_model': xgb_baseline if model_type == "hybrid_xgb_lstm" else None,
        }

        metrics = self._eval_metrics(model, val_loader, device)
        return metrics, model, test_ds

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_loss(model, loader, bce_fn_ins, bce_fn_tab, device, loss_weight_bce=1.0):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b in loader:
                p_ins, p_tab, p_im, p_tm = _forward_batch(model, b, device)

                y_ins = b['y_insulin'].to(device)
                y_tab = b['y_diabetes_tablet'].to(device)
                y_im = b['y_insulin_mask'].to(device)
                y_tm = b['y_diabetes_tablet_mask'].to(device)

                mse_part = masked_mse(p_ins, y_ins, y_im) + masked_mse(p_tab, y_tab, y_tm)
                bce_part = bce_fn_ins(p_im, y_im) + bce_fn_tab(p_tm, y_tm)
                loss = mse_part + loss_weight_bce * bce_part
                total += loss.item()
                n += 1
        return total / max(n, 1)

    @staticmethod
    def _compute_pos_weights(train_windows, num_ins, num_tabs, cap=15.0):
        """Per-drug-type pos_weight = num_negative / num_positive.

        Without this, the BCE loss treats "drug not prescribed" and "drug
        prescribed" equally.  Since most slots are 0 for any window, the
        model learns to predict all-zero masks.  pos_weight makes a missed
        prescription much costlier than a false positive.
        """
        ins_pos = torch.zeros(num_ins)
        tab_pos = torch.zeros(num_tabs)
        for w in train_windows:
            for mid, dose in w['insulin_doses_by_type'].items():
                idx = int(mid)
                if 0 <= idx < num_ins and dose > 0:
                    ins_pos[idx] += 1
            for mid, dose in w['drug_tablets_by_type'].items():
                idx = int(mid)
                if 0 <= idx < num_tabs and dose > 0:
                    tab_pos[idx] += 1

        n = float(len(train_windows))
        pw_ins = ((n - ins_pos) / ins_pos.clamp(min=1)).clamp(max=cap)
        pw_tab = ((n - tab_pos) / tab_pos.clamp(min=1)).clamp(max=cap)
        return pw_ins, pw_tab

    @staticmethod
    def _eval_metrics(model, loader, device):
        """
        Compute both classification metrics (drug selection) and
        regression metrics (dose accuracy) in a single pass.
        """
        model.eval()
        ys_ins, ps_ins, ms_ins = [], [], []
        ys_tab, ps_tab, ms_tab = [], [], []
        pm_ins, pm_tab = [], []

        with torch.no_grad():
            for b in loader:
                p_ins, p_tab, p_ins_m, p_tab_m = _forward_batch(model, b, device)

                ys_ins.append(b['y_insulin'])
                ps_ins.append(p_ins.cpu())
                ms_ins.append(b['y_insulin_mask'])
                pm_ins.append(p_ins_m.cpu())

                ys_tab.append(b['y_diabetes_tablet'])
                ps_tab.append(p_tab.cpu())
                ms_tab.append(b['y_diabetes_tablet_mask'])
                pm_tab.append(p_tab_m.cpu())

        y_i = torch.cat(ys_ins).numpy()
        p_i = torch.cat(ps_ins).numpy()
        m_i = torch.cat(ms_ins).numpy()
        y_t = torch.cat(ys_tab).numpy()
        p_t = torch.cat(ps_tab).numpy()
        m_t = torch.cat(ms_tab).numpy()

        pred_m_i = (torch.sigmoid(torch.cat(pm_ins)).numpy() > 0.5).astype(float)
        pred_m_t = (torch.sigmoid(torch.cat(pm_tab)).numpy() > 0.5).astype(float)

        # ── Classification: drug selection ─────────────────────────
        flat_true_i = m_i.flatten()
        flat_pred_i = pred_m_i.flatten()
        flat_true_t = m_t.flatten()
        flat_pred_t = pred_m_t.flatten()

        cls_ins = {
            'precision': precision_score(flat_true_i, flat_pred_i, zero_division=0),
            'recall': recall_score(flat_true_i, flat_pred_i, zero_division=0),
            'f1': f1_score(flat_true_i, flat_pred_i, zero_division=0),
            'accuracy': accuracy_score(flat_true_i, flat_pred_i),
        }
        cls_tab = {
            'precision': precision_score(flat_true_t, flat_pred_t, zero_division=0),
            'recall': recall_score(flat_true_t, flat_pred_t, zero_division=0),
            'f1': f1_score(flat_true_t, flat_pred_t, zero_division=0),
            'accuracy': accuracy_score(flat_true_t, flat_pred_t),
        }

        exact_match_ins = float((pred_m_i == m_i).all(axis=1).mean()) if m_i.ndim > 1 else 0.0
        exact_match_tab = float((pred_m_t == m_t).all(axis=1).mean()) if m_t.ndim > 1 else 0.0

        # ── Regression: dose accuracy (only on prescribed drugs) ──
        active_i = flat_true_i > 0
        active_t = flat_true_t > 0
        y_i_flat = y_i.flatten()
        p_i_flat = p_i.flatten()
        y_t_flat = y_t.flatten()
        p_t_flat = p_t.flatten()

        r2_ins = r2_score(y_i_flat[active_i], p_i_flat[active_i]) if active_i.any() else 0.0
        r2_tab = r2_score(y_t_flat[active_t], p_t_flat[active_t]) if active_t.any() else 0.0
        mae_ins = mean_absolute_error(y_i_flat[active_i], p_i_flat[active_i]) if active_i.any() else 0.0
        mae_tab = mean_absolute_error(y_t_flat[active_t], p_t_flat[active_t]) if active_t.any() else 0.0

        val_loss = (
            ((y_i_flat - p_i_flat) ** 2 * flat_true_i).sum() / max(flat_true_i.sum(), 1)
            + ((y_t_flat - p_t_flat) ** 2 * flat_true_t).sum() / max(flat_true_t.sum(), 1)
        )

        return {
            'val_loss': float(val_loss),
            # classification — insulin
            'prec_insulin': float(cls_ins['precision']),
            'recall_insulin': float(cls_ins['recall']),
            'f1_insulin': float(cls_ins['f1']),
            'acc_insulin': float(cls_ins['accuracy']),
            'exact_match_insulin': float(exact_match_ins),
            # classification — tablets
            'prec_tablets': float(cls_tab['precision']),
            'recall_tablets': float(cls_tab['recall']),
            'f1_tablets': float(cls_tab['f1']),
            'acc_tablets': float(cls_tab['accuracy']),
            'exact_match_tablets': float(exact_match_tab),
            # regression
            'r2_insulin': float(r2_ins),
            'r2_tablets': float(r2_tab),
            'mae_insulin': float(mae_ins),
            'mae_tablets': float(mae_tab),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, model, path="checkpoint.pt"):
        """Save model weights, architecture config, and all preprocessing
        artifacts needed for standalone inference (no DB required)."""
        data = self._data_cache
        arts = self._last_train_artifacts

        checkpoint = {
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'model_config': arts['model_config'],
            'preprocessing': {
                'scaler': arts['scaler'],
                'meas_mean': arts['meas_mean'],
                'meas_std': arts['meas_std'],
                'max_insulin': arts['max_insulin'],
                'max_tablet': arts['max_tablet'],
            },
            'mappings': {
                'unique_drugs': data['unique_drugs'],
                'unique_comorbities': data['unique_comorbities'],
                'insulin_names': self.repo.get_insulin_list(),
                'tablet_names': self.repo.get_tablets_list(),
                'drug_names': self.repo.get_additional_drugs_list(),
                'comorbidity_names': self.repo.get_comorbidities_list(),
            },
            'xgb_model': arts.get('xgb_model'),
        }
        torch.save(checkpoint, path)
        return path

    # ------------------------------------------------------------------
    # Search entry-points
    # ------------------------------------------------------------------

    def main(self, search_method="optuna", n_trials=30, model_type="gru",
             loss_weight_bce=1.0):
        """
        :param search_method: "grid" or "optuna"
        :param n_trials: number of Optuna trials (ignored for grid)
        :param model_type: "gru", "lstm", or "hybrid_xgb_lstm"
        :param loss_weight_bce: fixed weight for BCE loss relative to MSE
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
            best_artifacts = {}

            for hs, do, lr, bs in product(*grid.values()):
                metrics, model, test_ds = self.train_and_eval(
                    hs, do, lr, bs, model_type=model_type,
                    loss_weight_bce=loss_weight_bce)
                print(
                    f"hidden={hs}, do={do}, lr={lr}, bs={bs} | "
                    f"loss={metrics['val_loss']:.4f}  "
                    f"F1_ins={metrics['f1_insulin']:.3f}  F1_tab={metrics['f1_tablets']:.3f}  "
                    f"ExM_ins={metrics['exact_match_insulin']:.3f}  "
                    f"R²_ins={metrics['r2_insulin']:.4f}  MAE_ins={metrics['mae_insulin']:.4f}"
                )
                if metrics['val_loss'] < best_loss:
                    best_loss = metrics['val_loss']
                    best_config = dict(hidden_size=hs, dropout=do, lr=lr, batch_size=bs)
                    best_model = model
                    best_test_ds = test_ds
                    best_artifacts = dict(self._last_train_artifacts)

            self._last_train_artifacts = best_artifacts
            print("Best Grid Config:", best_config)

        elif search_method == "optuna":
            holder = {'model': None, 'test_ds': None, 'loss': float('inf'),
                      'artifacts': {}}

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
                    loss_weight_bce=loss_weight_bce,
                )
                if metrics['val_loss'] < holder['loss']:
                    holder.update(model=model, test_ds=test_ds,
                                  loss=metrics['val_loss'],
                                  artifacts=dict(self._last_train_artifacts))
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
            self._last_train_artifacts = holder['artifacts']

        # ── Final test-set evaluation ───────────────────────────────
        if best_model is not None and best_test_ds is not None and len(best_test_ds) > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_loader = DataLoader(
                best_test_ds, batch_size=32, shuffle=False,
                collate_fn=GlucoseDataset.collate_fn)
            test_metrics = self._eval_metrics(best_model, test_loader, device)
            print("\n=== TEST SET RESULTS ===")
            print(f"Combined Loss : {test_metrics['val_loss']:.4f}")
            print("── Drug Selection (Classification) ──")
            print(f"  Insulin  → Prec={test_metrics['prec_insulin']:.3f}  "
                  f"Rec={test_metrics['recall_insulin']:.3f}  "
                  f"F1={test_metrics['f1_insulin']:.3f}  "
                  f"ExactMatch={test_metrics['exact_match_insulin']:.3f}")
            print(f"  Tablets  → Prec={test_metrics['prec_tablets']:.3f}  "
                  f"Rec={test_metrics['recall_tablets']:.3f}  "
                  f"F1={test_metrics['f1_tablets']:.3f}  "
                  f"ExactMatch={test_metrics['exact_match_tablets']:.3f}")
            print("── Dose Accuracy (Regression) ──")
            print(f"  Insulin  → R²={test_metrics['r2_insulin']:.4f}  MAE={test_metrics['mae_insulin']:.4f}")
            print(f"  Tablets  → R²={test_metrics['r2_tablets']:.4f}  MAE={test_metrics['mae_tablets']:.4f}")

            self.save_checkpoint(best_model, "checkpoint.pt")
            print("Full checkpoint saved to checkpoint.pt")


if __name__ == '__main__':
    pipeline = FullModel()
    # Switch model_type here: "gru", "lstm", or "hybrid_xgb_lstm"
    pipeline.main(search_method="optuna", n_trials=15, model_type="lstm")
