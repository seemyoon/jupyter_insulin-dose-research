from itertools import zip_longest

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch

NUM_THERAPY_TYPES = 3


class GlucoseDataset(Dataset):

    def __init__(self, windows, static_dict, num_insulin, num_diabetes_tablets,
                 max_insulin, max_tablet, meas_mean=None, meas_std=None):
        """
        :param windows: list of dicts from MakeWindows.build_feature_windows
        :param static_dict: per-patient dict with keys:
            'static'     – Tensor of normalized static features
            'drug_idx'   – LongTensor of padded drug embedding indices
            'comorb_idx' – LongTensor of padded comorbidity embedding indices
            'xgb_preds'  – (optional) Tensor of XGBoost dose-baseline predictions
        :param max_insulin: max insulin dose (computed from training set only)
        :param max_tablet:  max tablet dose (computed from training set only)
        :param meas_mean: [cgm_mean, cbg_mean, ketone_mean] from training set
        :param meas_std:  [cgm_std, cbg_std, ketone_std] from training set
        """
        self.windows = windows
        self.static_dict = static_dict
        self.num_insulin = num_insulin
        self.num_diabetes_tablets = num_diabetes_tablets
        self.max_insulin = max(max_insulin, 1e-8)
        self.max_tablet = max(max_tablet, 1e-8)

        self.meas_mean = torch.tensor(meas_mean, dtype=torch.float32) if meas_mean else torch.zeros(3)
        self.meas_std = torch.tensor(meas_std, dtype=torch.float32) if meas_std else torch.ones(3)
        self.meas_std = self.meas_std.clamp(min=1e-8)

        first_pid = windows[0]['patient_id'] if windows else None
        self.has_xgb = first_pid is not None and 'xgb_preds' in self.static_dict.get(first_pid, {})

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        patient_id = window['patient_id']

        measurements = []
        for cgm, cbg, bk in zip_longest(
            window['cgm_values'], window['cbg_values'], window['blood_ketones'],
            fillvalue=0.0
        ):
            measurements.append([cgm, cbg, bk])

        meas_tensor = torch.tensor(measurements, dtype=torch.float32)
        meas_tensor = (meas_tensor - self.meas_mean) / self.meas_std

        patient_data = self.static_dict[patient_id]
        static = patient_data['static']
        drug_idx = patient_data['drug_idx']
        comorb_idx = patient_data['comorb_idx']

        food_intake = torch.tensor([window['food_intake_count']], dtype=torch.float32)

        therapy_onehot = torch.zeros(NUM_THERAPY_TYPES, dtype=torch.float32)
        therapy_val = window.get('therapy_type')
        if therapy_val is not None and 0 <= therapy_val < NUM_THERAPY_TYPES:
            therapy_onehot[therapy_val] = 1.0

        y_insulin = torch.zeros(self.num_insulin, dtype=torch.float32)
        for med_id, dose in window['insulin_doses_by_type'].items():
            if 0 <= int(med_id) < self.num_insulin:
                y_insulin[int(med_id)] = dose / self.max_insulin

        y_diabetes_tablet = torch.zeros(self.num_diabetes_tablets, dtype=torch.float32)
        for med_id, dose in window['drug_tablets_by_type'].items():
            if 0 <= int(med_id) < self.num_diabetes_tablets:
                y_diabetes_tablet[int(med_id)] = dose / self.max_tablet

        y_insulin_mask = (y_insulin > 0).float()
        y_diabetes_tablet_mask = (y_diabetes_tablet > 0).float()

        item = {
            'measurements': meas_tensor,
            'static': static,
            'drug_idx': drug_idx,
            'comorb_idx': comorb_idx,
            'food_intake': food_intake,
            'therapy_type': therapy_onehot,
            'y_insulin': y_insulin,
            'y_diabetes_tablet': y_diabetes_tablet,
            'y_insulin_mask': y_insulin_mask,
            'y_diabetes_tablet_mask': y_diabetes_tablet_mask,
        }

        if self.has_xgb:
            item['xgb_preds'] = patient_data['xgb_preds']

        return item

    @staticmethod
    def collate_fn(batch):
        """
        pad_sequence() aligns variable-length measurement sequences.
        torch.stack() batches fixed-size tensors.
        """
        measurements = [x['measurements'] for x in batch]
        measurements_pad = pad_sequence(measurements, batch_first=True)

        result = {
            'measurements': measurements_pad,
            'static': torch.stack([x['static'] for x in batch]),
            'drug_idx': torch.stack([x['drug_idx'] for x in batch]),
            'comorb_idx': torch.stack([x['comorb_idx'] for x in batch]),
            'food_intake': torch.stack([x['food_intake'] for x in batch]),
            'therapy_type': torch.stack([x['therapy_type'] for x in batch]),
            'y_insulin': torch.stack([x['y_insulin'] for x in batch]),
            'y_diabetes_tablet': torch.stack([x['y_diabetes_tablet'] for x in batch]),
            'y_insulin_mask': torch.stack([x['y_insulin_mask'] for x in batch]),
            'y_diabetes_tablet_mask': torch.stack([x['y_diabetes_tablet_mask'] for x in batch]),
        }

        if 'xgb_preds' in batch[0]:
            result['xgb_preds'] = torch.stack([x['xgb_preds'] for x in batch])

        return result
