"""
XGBoost baseline that predicts dose profiles from static patient features.

Used as the first stage in the Hybrid XGBoost+LSTM architecture:
  1. XGBoost learns a patient-level dose baseline from tabular features.
  2. Its predictions are fed as extra input to the LSTM, which refines
     them using temporal patterns (CGM trends, meal timing, etc.).
"""

from collections import defaultdict

import numpy as np

try:
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:
    xgb = None
    MultiOutputRegressor = None


class XGBoostBaseline:
    def __init__(self, num_insulin, num_tablets):
        if xgb is None:
            raise ImportError(
                "xgboost and scikit-learn are required for the hybrid model. "
                "Install them with: pip install xgboost scikit-learn"
            )
        self.num_insulin = num_insulin
        self.num_tablets = num_tablets
        self.model = None

    @property
    def output_dim(self):
        return self.num_insulin + self.num_tablets

    def _patient_avg_doses(self, windows, max_insulin, max_tablet):
        """Average normalized dose per patient across all their windows."""
        ins_accum = defaultdict(list)
        tab_accum = defaultdict(list)

        for w in windows:
            pid = w['patient_id']

            ins_vec = np.zeros(self.num_insulin)
            for mid, dose in w['insulin_doses_by_type'].items():
                if 0 <= int(mid) < self.num_insulin:
                    ins_vec[int(mid)] = dose / max(max_insulin, 1e-8)
            ins_accum[pid].append(ins_vec)

            tab_vec = np.zeros(self.num_tablets)
            for mid, dose in w['drug_tablets_by_type'].items():
                if 0 <= int(mid) < self.num_tablets:
                    tab_vec[int(mid)] = dose / max(max_tablet, 1e-8)
            tab_accum[pid].append(tab_vec)

        targets = {}
        for pid in set(ins_accum) | set(tab_accum):
            ins_avg = np.mean(ins_accum.get(pid, [np.zeros(self.num_insulin)]), axis=0)
            tab_avg = np.mean(tab_accum.get(pid, [np.zeros(self.num_tablets)]), axis=0)
            targets[pid] = np.concatenate([ins_avg, tab_avg])
        return targets

    def fit(self, train_windows, static_dict, max_insulin, max_tablet):
        """
        Train XGBoost on training patients' static features → mean dose profile.

        :param train_windows: training-set windows only
        :param static_dict: {pid: {'static': Tensor, ...}} for feature lookup
        :param max_insulin: training max insulin dose (for normalization)
        :param max_tablet: training max tablet dose (for normalization)
        """
        targets = self._patient_avg_doses(train_windows, max_insulin, max_tablet)
        pids = sorted(targets.keys())

        X = np.array([static_dict[pid]['static'].numpy() for pid in pids])
        y = np.array([targets[pid] for pid in pids])

        self.model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
            )
        )
        self.model.fit(X, y)
        return self

    def predict_all(self, static_dict):
        """
        Return {patient_id: np.array[output_dim]} predictions for every patient.
        """
        pids = sorted(static_dict.keys())
        X = np.array([static_dict[pid]['static'].numpy() for pid in pids])
        preds = self.model.predict(X)
        return {pid: preds[i] for i, pid in enumerate(pids)}
