from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch


class GlucoseDataset(Dataset):
    # PyTorch must somehow obtain data (one example at a time) when training a model.
    # To do this, PyTorch uses a special class called Dataset.

    # The model does not learn from one example at a time, it learns from batches (packs).
    # Why not submit everything at once?
    # Because:
    # -you may have millions of examples → they won't fit in memory
    # -training the entire sample at once is slow and unstable
    # -the gradients would be unstable

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

    def __len__(self):
        return len(self.windows)

        # When a DataLoader (in main) object is created, it automatically:
        # - calls __len__() on the dataset to determine the total number of examples,
        # - then calls __getitem__(idx) in a loop for each example from the batch (a batch is a group of examples),
        # - idx is simply the example number, for example: 0, 1, 2, and so on up to len(dataset) - 1.

    def __getitem__(self, idx):
        # idx is simply the window index: 0, 1, 2, ..., len(windows)-1
        # w is a single window (e.g., for 1 day)
        # pid = w['patient_id'] extracts the ID of the patient to whom this window belongs

        window = self.windows[idx]
        patient_id = window['patient_id']

        measurements = []

        for cgm, cbg, blood_ketones in zip(window['cgm_values'], window['cbg_values'], window['blood_ketones']):
            measurements.append([cgm, cbg, blood_ketones])

        measurements_with_tensors = torch.tensor(measurements, dtype=torch.float32)
        # converts a list of lists seq (where each nested list is [cgm, cbg, blood_ketone] for a single time step) into a single tensor with dimensions [T, 3], where T is the number of time steps.

        # Example:
        # seq = [
        #   [5.2, 5.0, 0.1],
        #   [5.6, 5.5, 0.15],
        #   [6.1, 6.0, 0.12],
        # ]

        # dynamic = torch.tensor(seq, dtype=torch.float32)
        # print(dynamic.shape)  # torch.Size([3, 3])

        static = self.static_dict[patient_id]

        food_intake = torch.tensor([window['food_intake_count']], dtype=torch.float32)

        y_insulin = torch.zeros(self.num_insulin, dtype=torch.float32)
        for med_id, dose in window['insulin_doses_by_type'].items():
            if 0 <= med_id < self.num_insulin:
                y_insulin[med_id] = dose

        y_diabetes_tablet = torch.zeros(self.num_diabetes_tablets, dtype=torch.float32)
        for med_id, dose in window['drug_tablets_by_type'].items():
            if 0 <= med_id < self.num_diabetes_tablets:
                y_diabetes_tablet[med_id] = dose

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

        # Collects dynamic data from all examples into a list.
        # measurements = [
        #    tensor([[5.2, 5.0, 0.1], [5.6, 5.5, 0.15]]),    # example 1 with 2 time points
        #    tensor([[6.1, 6.0, 0.12]])                      # example 2 with 1 time point
        # ]

        measurements_pad = pad_sequence(measurements, batch_first=True)

        # Pad (fill) the sequences so that they all have the same length in time (T).
        # dyn_pad = tensor([
        #     [[5.2, 5.0, 0.1], [5.6, 5.5, 0.15]],
        #     [[6.1, 6.0, 0.12], [0.0, 0.0, 0.0]]
        # ])  # size [2, 2, 3] — 2 examples, maximum 2 time points, 3 features (cgm, cbg, ketone)

        statics = torch.stack([x['static'] for x in batch])
        food_intake = torch.stack([x['food_intake'] for x in batch])
        y_insulin = torch.stack([x['y_insulin'] for x in batch])
        y_diabetes_tablet = torch.stack([x['y_diabetes_tablet'] for x in batch])

        # todo why exactly statics, food_intake, y_insulin, y_diabetes_tablet

        return {
            'measurements': measurements_pad,
            'static': statics,
            'food_intake': food_intake,
            'y_insulin': y_insulin,
            'y_diabetes_tablet': y_diabetes_tablet
        }
