import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
import optuna


class XGBoostRealization:
    def __init__(
            self,
            train_loader,
            val_loader,
            test_loader,
            num_ins,
            num_drug_types,
            optimize_with_optuna=True,
            n_trials=20,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_ins = num_ins
        self.num_drug_types = num_drug_types
        self.optimize_with_optuna = optimize_with_optuna
        self.n_trials = n_trials

        self.models = {
            "ins_dose": None,
            "tab_dose": None,
            "ins_mask": None,
            "tab_mask": None,
        }

    def _prepare_numpy_data(self, loader):
        X_list, y_ins_list, y_tab_list, y_ins_mask_list, y_tab_mask_list = [], [], [], [], []

        for batch in loader:
            measurements = batch["measurements"]
            static = batch["static"]
            food = batch["food_intake"]

            y_insulin = batch["y_insulin"]
            y_tab = batch["y_diabetes_tablet"]
            y_ins_mask = batch["y_insulin_mask"]
            y_tab_mask = batch["y_diabetes_tablet_mask"]

            B, T, F = measurements.shape
            dynamic_feat = measurements.reshape(B, -1)
            x = torch.cat([dynamic_feat, static, food], dim=1).numpy()

            X_list.append(x)
            y_ins_list.append(y_insulin.numpy())
            y_tab_list.append(y_tab.numpy())
            y_ins_mask_list.append(y_ins_mask.numpy())
            y_tab_mask_list.append(y_tab_mask.numpy())

        X = np.concatenate(X_list, axis=0)
        y_ins = np.concatenate(y_ins_list, axis=0)
        y_tab = np.concatenate(y_tab_list, axis=0)
        y_ins_mask = np.concatenate(y_ins_mask_list, axis=0)
        y_tab_mask = np.concatenate(y_tab_mask_list, axis=0)
        return X, y_ins, y_tab, y_ins_mask, y_tab_mask

    def _objective(self, trial, X_train, y_train, X_val, y_val, task_type="regression"):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "random_state": 42,
            "verbosity": 0,
        }

        match task_type:
            case "regression":
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                return mean_squared_error(y_val, preds)
            case "classification":
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict_proba(X_val)[:, 1]
                return log_loss(y_val, preds)

    def _tune_model(self, X_train, y_train, X_val, y_val, task_type):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_val, y_val, task_type),
                       n_trials=self.n_trials)
        print(f"best params for {task_type}: {study.best_trial.params}")
        return study.best_trial.params

    def train(self):
        print("Preparing data for XGBoost + Optuna...")
        X_train, y_ins_train, y_tab_train, y_ins_mask_train, y_tab_mask_train = self._prepare_numpy_data(
            self.train_loader)
        X_val, y_ins_val, y_tab_val, y_ins_mask_val, y_tab_mask_val = self._prepare_numpy_data(self.val_loader)

        if self.optimize_with_optuna:
            print("Starting hyperparameter optimization with Optuna...")

            best_params_reg = self._tune_model(X_train, y_ins_train, X_val, y_ins_val, "regression")
            best_params_clf = self._tune_model(X_train, y_ins_mask_train, X_val, y_ins_mask_val, "classification")
        else:
            best_params_reg = {
                "learning_rate": 0.1, "max_depth": 6, "n_estimators": 400,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                "gamma": 0, "reg_lambda": 1, "random_state": 42,
            }
            best_params_clf = best_params_reg.copy()

        print("Training final models with best params...")

        self.models["ins_dose"] = xgb.XGBRegressor(**best_params_reg)
        self.models["tab_dose"] = xgb.XGBRegressor(**best_params_reg)
        self.models["ins_mask"] = xgb.XGBClassifier(**best_params_clf, use_label_encoder=False, eval_metric="logloss")
        self.models["tab_mask"] = xgb.XGBClassifier(**best_params_clf, use_label_encoder=False, eval_metric="logloss")

        self.models["ins_dose"].fit(X_train, y_ins_train, eval_set=[(X_val, y_ins_val)], verbose=False)
        self.models["tab_dose"].fit(X_train, y_tab_train, eval_set=[(X_val, y_tab_val)], verbose=False)
        self.models["ins_mask"].fit(X_train, y_ins_mask_train, eval_set=[(X_val, y_ins_mask_val)], verbose=False)
        self.models["tab_mask"].fit(X_train, y_tab_mask_train, eval_set=[(X_val, y_tab_mask_val)], verbose=False)

        print("Training completed.")
        self.test()

    def test(self):
        print("\nTesting XGBoost models...")
        X_test, y_ins_test, y_tab_test, y_ins_mask_test, y_tab_mask_test = self._prepare_numpy_data(self.test_loader)

        preds = {
            "ins_dose": self.models["ins_dose"].predict(X_test),
            "tab_dose": self.models["tab_dose"].predict(X_test),
            "ins_mask": self.models["ins_mask"].predict_proba(X_test)[:, 1],
            "tab_mask": self.models["tab_mask"].predict_proba(X_test)[:, 1],
        }

        mse_ins = mean_squared_error(y_ins_test, preds["ins_dose"])
        mse_tab = mean_squared_error(y_tab_test, preds["tab_dose"])
        bce_ins = log_loss(y_ins_mask_test, preds["ins_mask"])
        bce_tab = log_loss(y_tab_mask_test, preds["tab_mask"])
        acc_ins = accuracy_score(y_ins_mask_test, (preds["ins_mask"] > 0.5).astype(int))
        acc_tab = accuracy_score(y_tab_mask_test, (preds["tab_mask"] > 0.5).astype(int))

        print(f"Test Results:")
        print(f"  Insulin Dose MSE: {mse_ins:.4f}")
        print(f"  Tablet Dose  MSE: {mse_tab:.4f}")
        print(f"  Insulin Mask BCE: {bce_ins:.4f}, Acc: {acc_ins:.4f}")
        print(f"  Tablet  Mask BCE: {bce_tab:.4f}, Acc: {acc_tab:.4f}")
