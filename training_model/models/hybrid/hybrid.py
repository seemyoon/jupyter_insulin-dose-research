import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.dummy import DummyRegressor, DummyClassifier

from training_model.models.lstm_linear.lstm_linear import LSTMLinearModel


class HybridRealization:
    def __init__(
            self,
            num_drug_types,
            num_ins,
            train_loader,
            val_loader,
            test_loader,
            static_dim=64,
            hidden_size=64,
            epochs=200,
            batch_size=32,
            lr=1e-3,
            patience=10,
            optuna_trials=5,
            xgb_train_on_train_val=True
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.num_drug_types = num_drug_types
        self.num_ins = num_ins
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMLinearModel(
            static_dim=static_dim,
            hidden_size=hidden_size,
            num_drug_types=num_drug_types,
            num_insulin_types=num_ins
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss_fn = nn.MSELoss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_path = 'best_model_lstm.pth'

        self.optuna_trials = optuna_trials
        self.xgb_train_on_train_val = xgb_train_on_train_val
        self.xgb_models = None

    def train_lstm(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_mse_loss, train_bce_loss = 0.0, 0.0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food_intake = batch['food_intake'].to(self.device)

                y_insulin = batch['y_insulin'].to(self.device)
                y_tab = batch['y_diabetes_tablet'].to(self.device)
                y_ins_mask = batch['y_insulin_mask'].to(self.device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(self.device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = self.model(
                    measurements,
                    static,
                    food_intake
                )

                mse_loss = self.mse_loss_fn(pred_ins_dose, y_insulin) + self.mse_loss_fn(pred_tab_dose, y_tab)
                bce_loss = self.bce_loss_fn(pred_ins_mask, y_ins_mask) + self.bce_loss_fn(pred_tab_mask, y_tab_mask)
                loss = mse_loss + bce_loss

                loss.backward(retain_graph=True)
                self.optimizer.step()

                train_mse_loss += mse_loss.item()
                train_bce_loss += bce_loss.item()

            train_mse_loss /= len(self.train_loader)
            train_bce_loss /= len(self.train_loader)

            val_mse_loss, val_bce_loss = self.validate_lstm()

            print(f"[LSTM] Epoch {epoch:3d} | Train MSE: {train_mse_loss:.4f}, BCE: {train_bce_loss:.4f} | "
                  f"Val MSE: {val_mse_loss:.4f}, BCE: {val_bce_loss:.4f}")

            val_total_loss = val_mse_loss + val_bce_loss
            if val_total_loss < self.best_val_loss:
                self.best_val_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping LSTM triggered after {epoch + 1} epochs.")
                break

        print("Training finished.")

    def validate_lstm(self):
        self.model.eval()
        val_mse_loss, val_bce_loss = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food_intake = batch['food_intake'].to(self.device)

                y_insulin = batch['y_insulin'].to(self.device)
                y_tab = batch['y_diabetes_tablet'].to(self.device)
                y_ins_mask = batch['y_insulin_mask'].to(self.device)
                y_tab_mask = batch['y_diabetes_tablet_mask'].to(self.device)

                pred_ins_dose, pred_tab_dose, pred_ins_mask, pred_tab_mask = self.model(measurements, static,
                                                                                        food_intake)

                val_mse_loss += self.mse_loss_fn(pred_ins_dose, y_insulin).item() + self.mse_loss_fn(pred_tab_dose,
                                                                                                     y_tab).item()
                val_bce_loss += self.bce_loss_fn(pred_ins_mask, y_ins_mask).item() + self.bce_loss_fn(pred_tab_mask,
                                                                                                      y_tab_mask).item()

        return val_mse_loss / len(self.val_loader), val_bce_loss / len(self.val_loader)

    def extract_embeddings(self, loader):
        X_list, y_ins_list, y_tab_list, y_ins_mask_list, y_tab_mask_list = [], [], [], [], []

        self.model.eval()

        with torch.no_grad():
            for batch in loader:

                measurements = batch['measurements'].to(self.device)
                static = batch['static'].to(self.device)
                food = batch['food_intake'].to(self.device)

                feat = self.model.extract_features(measurements, static, food)
                feat_np = feat.cpu().numpy()

                X_batch = feat_np

                y_ins = batch.get('y_insulin')
                y_tab = batch.get('y_diabetes_tablet')
                y_ins_mask = batch.get('y_insulin_mask')
                y_tab_mask = batch.get('y_diabetes_tablet_mask')

                if y_ins is not None:
                    y_ins_list.append(y_ins.cpu().numpy())
                    y_tab_list.append(y_tab.cpu().numpy())
                    y_ins_mask_list.append(y_ins_mask.cpu().numpy())
                    y_tab_mask_list.append(y_tab_mask.cpu().numpy())

                X_list.append(X_batch)

        X = np.concatenate(X_list, axis=0)
        if len(y_ins_list) > 0:
            y_ins = np.concatenate(y_ins_list, axis=0)
            y_tab = np.concatenate(y_tab_list, axis=0)
            y_ins_mask = np.concatenate(y_ins_mask_list, axis=0)
            y_tab_mask = np.concatenate(y_tab_mask_list, axis=0)
        else:
            y_ins = y_tab = y_ins_mask = y_tab_mask = None

        return X, y_ins, y_tab, y_ins_mask, y_tab_mask

    @staticmethod
    def _xgb_objective(trial, X_tr, y_tr, X_val, y_val, task='reg'):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbosity": 0,
        }
        if task == 'reg':
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)
        else:
            params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'base_score': 0.5
            })
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds_proba = model.predict_proba(X_val)
            return log_loss(y_val, preds_proba)

    def tune_xgb(
            self,
            X_train,
            y_train_ins,
            y_train_tab,
            y_train_ins_mask,
            y_train_tab_mask,

            X_val,
            y_val_ins,
            y_val_tab,
            y_val_ins_mask,
            y_val_tab_mask
    ):
        y_train_ins_mask = np.nan_to_num(y_train_ins_mask, nan=0).astype(np.int32)
        y_val_ins_mask = np.nan_to_num(y_val_ins_mask, nan=0).astype(np.int32)
        y_train_tab_mask = np.nan_to_num(y_train_tab_mask, nan=0).astype(np.int32)
        y_val_tab_mask = np.nan_to_num(y_val_tab_mask, nan=0).astype(np.int32)

        best_params = {}

        print("Tuning XGBoost for insulin REGRESSORS...")
        best_params['reg_ins'] = []
        for i in range(y_train_ins.shape[1]):
            y_tr_col, y_val_col = y_train_ins[:, i], y_val_ins[:, i]
            if np.all(y_tr_col == 0):
                print(f"Skip insulin dose {i} (all zeros)")
                best_params['reg_ins'].append(None)
                continue
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda t: self._xgb_objective(t, X_train, y_tr_col, X_val, y_val_col, task='reg'),
                n_trials=self.optuna_trials
            )
            best_params['reg_ins'].append(study.best_trial.params)
            print(f"Best insulin reg params for drug {i}: {study.best_trial.params}")

        print("\nTuning XGBoost for tablet REGRESSORS...")
        best_params['reg_tab'] = []
        for i in range(y_train_tab.shape[1]):
            y_tr_col, y_val_col = y_train_tab[:, i], y_val_tab[:, i]
            if np.all(y_tr_col == 0):
                print(f"Skip tablet dose {i} (all zeros)")
                best_params['reg_tab'].append(None)
                continue
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda t: self._xgb_objective(t, X_train, y_tr_col, X_val, y_val_col, task='reg'),
                n_trials=self.optuna_trials
            )
            best_params['reg_tab'].append(study.best_trial.params)
            print(f"Best tablet reg params for drug {i}: {study.best_trial.params}")

        print("\nTuning XGBoost for insulin CLASSIFIERS...")
        best_params['clf_ins'] = []

        print('y_train_ins_mask.shape[1]: ', y_train_ins_mask.shape[1])
        for i in range(y_train_ins_mask.shape[1]):
            y_tr_col, y_val_col = y_train_ins_mask[:, i], y_val_ins_mask[:, i]
            if len(np.unique(y_tr_col)) < 2:
                print(f"Skip insulin mask {i} (only one class: {np.unique(y_tr_col)})")
                best_params['clf_ins'].append(None)
                continue
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda t: self._xgb_objective(t, X_train, y_tr_col, X_val, y_val_col, task='clf'),
                n_trials=self.optuna_trials
            )
            best_params['clf_ins'].append(study.best_trial.params)
            print(f"Best insulin clf params for drug {i}: {study.best_trial.params}")

        print("\nTuning XGBoost for tablet CLASSIFIERS...")
        best_params['clf_tab'] = []
        for i in range(y_train_tab_mask.shape[1]):
            y_tr_col, y_val_col = y_train_tab_mask[:, i], y_val_tab_mask[:, i]
            if len(np.unique(y_tr_col)) < 2:
                print(f"Skip tablet mask {i} (only one class: {np.unique(y_tr_col)})")
                best_params['clf_tab'].append(None)
                continue
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda t: self._xgb_objective(t, X_train, y_tr_col, X_val, y_val_col, task='clf'),
                n_trials=self.optuna_trials
            )
            best_params['clf_tab'].append(study.best_trial.params)
            print(f"Best tablet clf params for drug {i}: {study.best_trial.params}")

        return best_params

    def train_xgb_final(
            self,
            best_params,
            X_train,
            y_train_ins,
            y_train_tab,
            y_train_ins_mask,
            y_train_tab_mask,
            X_val,
            y_val_ins,
            y_val_tab,
            y_val_ins_mask,
            y_val_tab_mask
    ):

        if self.xgb_train_on_train_val:
            X_fit = np.vstack([X_train, X_val])
            y_fit_ins = np.vstack([y_train_ins, y_val_ins])
            y_fit_tab = np.vstack([y_train_tab, y_val_tab])
            y_fit_ins_mask = np.vstack([y_train_ins_mask, y_val_ins_mask])
            y_fit_tab_mask = np.vstack([y_train_tab_mask, y_val_tab_mask])
        else:
            X_fit = X_train
            y_fit_ins = y_train_ins
            y_fit_tab = y_train_tab
            y_fit_ins_mask = y_train_ins_mask
            y_fit_tab_mask = y_train_tab_mask

        models = {}

        models['ins_reg'] = []
        if best_params.get('reg_ins'):
            for i, p in enumerate(best_params['reg_ins']):
                y_col = y_fit_ins[:, i]
                if p is None or np.all(y_col == 0):
                    print(f"Skip insulin regressor {i} (all zeros) -> using DummyRegressor")
                    dummy = DummyRegressor(strategy="constant", constant=0.0)
                    dummy.fit(X_fit, y_col)
                    models['ins_reg'].append(dummy)
                    continue

                print(f"Training insulin regressor {i}...")
                try:
                    reg = xgb.XGBRegressor(**p)
                    reg.fit(X_fit, y_col)
                    models['ins_reg'].append(reg)
                except Exception as e:
                    print(f"XGBRegressor {i} failed: {e}, using DummyRegressor")
                    dummy = DummyRegressor(strategy="mean")
                    dummy.fit(X_fit, y_col)
                    models['ins_reg'].append(dummy)
        else:
            models['ins_reg'] = None

        models['tab_reg'] = []
        if best_params.get('reg_tab'):
            for i, p in enumerate(best_params['reg_tab']):
                y_col = y_fit_tab[:, i]
                if p is None or np.all(y_col == 0):
                    print(f"Skip tablet regressor {i} (all zeros) -> using DummyRegressor")
                    dummy = DummyRegressor(strategy="constant", constant=0.0)
                    dummy.fit(X_fit, y_col)
                    models['tab_reg'].append(dummy)
                    continue

                print(f"Training tablet regressor {i}...")
                try:
                    reg = xgb.XGBRegressor(**p)
                    reg.fit(X_fit, y_col)
                    models['tab_reg'].append(reg)
                except Exception as e:
                    print(f"XGBRegressor {i} failed: {e}, using DummyRegressor")
                    dummy = DummyRegressor(strategy="mean")
                    dummy.fit(X_fit, y_col)
                    models['tab_reg'].append(dummy)
        else:
            models['tab_reg'] = None

        models['ins_clf'] = []
        if best_params.get('clf_ins'):
            for i, p in enumerate(best_params['clf_ins']):
                y_col = y_fit_ins_mask[:, i]
                unique_labels = np.unique(y_col)

                if p is None or len(unique_labels) < 2:
                    print(f"Skip insulin classifier {i} (only one class) -> using DummyClassifier")
                    dummy = DummyClassifier(strategy="most_frequent")
                    dummy.fit(X_fit, y_col)
                    dummy.classes_ = np.array([0, 1])
                    models['ins_clf'].append(dummy)
                    continue

                print(f"Training insulin classifier {i}...")
                try:
                    p = p.copy()
                    p.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'base_score': 0.5})

                    clf = xgb.XGBClassifier(**p)

                    y_val_col = y_val_ins_mask[:, i]
                    if len(np.unique(y_val_col)) > 1:
                        clf.fit(X_fit, y_col, eval_set=[(X_val, y_val_col)], verbose=False)
                    else:
                        clf.fit(X_fit, y_col)

                    models['ins_clf'].append(clf)
                except Exception as e:
                    print(f"XGBClassifier {i} failed: {e}, using DummyClassifier")
                    dummy = DummyClassifier(strategy="most_frequent")
                    dummy.fit(X_fit, y_col)
                    dummy.classes_ = np.array([0, 1])
                    models['ins_clf'].append(dummy)
        else:
            models['ins_clf'] = None

        models['tab_clf'] = []
        if best_params.get('clf_tab'):
            for i, p in enumerate(best_params['clf_tab']):
                y_col = y_fit_tab_mask[:, i]
                unique_labels = np.unique(y_col)

                if p is None or len(unique_labels) < 2:
                    print(f"Skip tablet classifier {i} (only one class) -> using DummyClassifier")
                    dummy = DummyClassifier(strategy="most_frequent")
                    dummy.fit(X_fit, y_col)
                    dummy.classes_ = np.array([0, 1])
                    models['tab_clf'].append(dummy)
                    continue

                print(f"Training tablet classifier {i}...")
                try:
                    p = p.copy()
                    p.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'base_score': 0.5})

                    clf = xgb.XGBClassifier(**p)
                    y_val_col = y_val_tab_mask[:, i]

                    if len(np.unique(y_val_col)) > 1:
                        clf.fit(X_fit, y_col, eval_set=[(X_val, y_val_col)], verbose=False)
                    else:
                        clf.fit(X_fit, y_col)

                    models['tab_clf'].append(clf)
                except Exception as e:
                    print(f"XGBClassifier {i} failed: {e}, using DummyClassifier")
                    dummy = DummyClassifier(strategy="most_frequent")
                    dummy.fit(X_fit, y_col)
                    dummy.classes_ = np.array([0, 1])
                    models['tab_clf'].append(dummy)
        else:
            models['tab_clf'] = None

        self.xgb_models = models
        return models

    @staticmethod
    def predict_proba_multi_models(model_list, X):
        preds = []
        for m in model_list:
            if m is None:
                preds.append(np.zeros((X.shape[0],)))
            else:
                preds.append(m.predict_proba(X)[:, 1])
        return np.column_stack(preds)

    @staticmethod
    def predict_multi_models(model_list, X):
        preds = []
        for m in model_list:
            if m is None:
                preds.append(np.zeros((X.shape[0],)))
            else:
                preds.append(m.predict(X))
        return np.column_stack(preds)

    def eval_xgb(self, X_test, y_test_ins, y_test_tab, y_test_ins_mask, y_test_tab_mask):
        results = {}

        ins_pred = self.predict_multi_models(self.xgb_models['ins_reg'], X_test)
        tab_pred = self.predict_multi_models(self.xgb_models['tab_reg'], X_test)
        results['mse_ins'] = mean_squared_error(y_test_ins, ins_pred)
        results['mse_tab'] = mean_squared_error(y_test_tab, tab_pred)

        ins_prob_arr = self.predict_proba_multi_models(self.xgb_models['ins_clf'], X_test)
        tab_prob_arr = self.predict_proba_multi_models(self.xgb_models['tab_clf'], X_test)

        results['bce_ins'] = log_loss(y_test_ins_mask.reshape(-1), ins_prob_arr.reshape(-1))
        results['bce_tab'] = log_loss(y_test_tab_mask.reshape(-1), tab_prob_arr.reshape(-1))

        results['acc_ins'] = accuracy_score(y_test_ins_mask.reshape(-1), (ins_prob_arr.reshape(-1) > 0.5).astype(int))
        results['acc_tab'] = accuracy_score(y_test_tab_mask.reshape(-1), (tab_prob_arr.reshape(-1) > 0.5).astype(int))

        return results

    def hybrid_train(self):
        self.train_lstm()
        self.model.load_state_dict(torch.load(self.best_model_path))

        X_train, y_train_ins, y_train_tab, y_train_ins_mask, y_train_tab_mask = self.extract_embeddings(
            self.train_loader
        )

        X_val, y_val_ins, y_val_tab, y_val_ins_mask, y_val_tab_mask = self.extract_embeddings(
            self.val_loader
        )

        X_test, y_test_ins, y_test_tab, y_test_ins_mask, y_test_tab_mask = self.extract_embeddings(
            self.test_loader
        )

        print('TUNING XGB')
        best_params = self.tune_xgb(
            X_train,
            y_train_ins,
            y_train_tab,
            y_train_ins_mask,
            y_train_tab_mask,

            X_val,
            y_val_ins,
            y_test_tab,
            y_val_ins_mask,
            y_val_tab_mask
        )

        self.train_xgb_final(
            best_params,
            X_train,
            y_train_ins,
            y_train_tab,
            y_train_ins_mask,
            y_train_tab_mask,

            X_val,
            y_val_ins,
            y_test_tab,
            y_test_ins_mask,
            y_test_tab_mask

        )

        results = self.eval_xgb(
            X_test,
            y_test_ins,
            y_test_tab,
            y_test_ins_mask,
            y_test_tab_mask
        )

        print("Hybrid test results:", results)
        return results

    def hybrid_predict(self, new_loader):
        if self.xgb_models is None:
            raise RuntimeError("Train hybrid first (hybrid_train) or load xgb models")

        X_new, _, _, _, _ = self.extract_embeddings(new_loader)
        ins_pred = self.xgb_models['ins_reg'].predict(X_new)
        tab_pred = self.xgb_models['tab_reg'].predict(X_new)
        ins_prob = self.xgb_models['ins_clf'].predict_proba(X_new)
        tab_prob = self.xgb_models['tab_clf'].predict_proba(X_new)

        ins_prob_arr = np.vstack([p[:, 1] for p in ins_prob]).T
        tab_prob_arr = np.vstack([p[:, 1] for p in tab_prob]).T

        return {
            'insulin_dose_pred': ins_pred,
            'tablet_dose_pred': tab_pred,
            'insulin_prob': ins_prob_arr,
            'tablet_prob': tab_prob_arr
        }
