from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch


class GlucoseDataset(Dataset):

    def __init__(self, windows, static_dict, num_insulin, num_diabetes_tablets):
        """

        :param windows:list of dictionaries returned by MakeWindows.build_feature_windows,
            where for each window there is:
            - insulin_doses_by_type: {id: dose}
            - drug_tablets_by_type: {id: dose}
            - cgm_values, cbg_values, blood_ketones: lists of floats
            - food_intake_count: int
            - therapy_type: enum or None
            - patient_id
        :param static_dict: {patient_id: Tensor[static_dim]}
        """
        self.windows = windows
        self.static_dict = static_dict
        self.num_insulin = num_insulin
        self.num_diabetes_tablets = num_diabetes_tablets

        self.max_insulin = max([dose for window in windows for dose in window['insulin_doses_by_type'].values()])
        self.max_tablet = max([dose for window in windows for dose in window['drug_tablets_by_type'].values()])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        window = self.windows[idx]
        patient_id = window['patient_id']

        measurements = []

        for cgm, cbg, blood_ketones in zip(window['cgm_values'], window['cbg_values'], window['blood_ketones']):
            measurements.append([cgm, cbg, blood_ketones])

        measurements_with_tensors = torch.tensor(measurements, dtype=torch.float32)

        static = self.static_dict[patient_id]

        food_intake = torch.tensor([window['food_intake_count']], dtype=torch.float32)

        y_insulin = torch.zeros(self.num_insulin, dtype=torch.float32)
        for med_id, dose in window['insulin_doses_by_type'].items():
            if 0 <= int(med_id) < self.num_insulin:
                y_insulin[int(med_id)] = dose / self.max_insulin

        y_diabetes_tablet = torch.zeros(self.num_diabetes_tablets, dtype=torch.float32)
        for med_id, dose in window['drug_tablets_by_type'].items():
            if 0 <= int(med_id) < self.num_diabetes_tablets:
                y_diabetes_tablet[int(med_id)] = dose / self.max_tablet

        return {
            'measurements': measurements_with_tensors,
            'static': static,
            'food_intake': food_intake,
            'y_insulin': y_insulin,
            'y_diabetes_tablet': y_diabetes_tablet
        }

    @staticmethod
    def collate_fn(batch):

        """
        torch.stack()    Stacks tensors of the same size into a batch
        pad_sequence()    Align tensors of different lengths in time
        batch_first=True    The batch is the first dimension [B, T, D]

        """
        measurements = [x['measurements'] for x in batch]

        measurements_pad = pad_sequence(measurements, batch_first=True)

        statics = torch.stack([x['static'] for x in batch])
        food_intake = torch.stack([x['food_intake'] for x in batch])
        y_insulin = torch.stack([x['y_insulin'] for x in batch])
        y_diabetes_tablet = torch.stack([x['y_diabetes_tablet'] for x in batch])

        return {
            'measurements': measurements_pad,
            'static': statics,
            'food_intake': food_intake,
            'y_insulin': y_insulin,
            'y_diabetes_tablet': y_diabetes_tablet
        }
