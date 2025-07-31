import torch
import torch.nn as nn


class DiabetesModel(nn.Module):
    def __init__(self, static_dim, hidden_size, num_insulin_types, num_drug_types):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)  # 3 - cgm, cgb, blood_ketone
        self.static_data = nn.Linear(static_dim, 32)
        self.food_intake = nn.Linear(1, 32)

        self.combined_layers = nn.Linear(hidden_size + 32 + 32, 64)
        # This is the layer that combines all three sources of information:
        # h from LSTM → size [B, hidden_size]
        # s from fc_static → [B,32]
        # m from fc_meals → [B,32]

        self.dose_insulin = nn.Linear(64, num_insulin_types)
        # The output is a vector of doses for all insulins — the length of num_insulin_types.
        # Each element is the dose of a specific drug.
        self.dose_drug_tabletes = nn.Linear(64, num_drug_types)

        self.relu = nn.ReLU()

    def forward(self, dynamic_data, static_data, food_intake):
        # print('dynamic_data shape: ', dynamic_data.shape)
        # print('static_data shape: ', static_data.shape)
        # print('food_intake shape: ', food_intake.shape)

        _, (h_n) = self.lstm(
            dynamic_data)  # dynamic_data — [B, T, 3] B - Batch size (f.e 32),Time steps (f.e 24), 3 (quantity of features: CGM, CBG, blood_ketones)
        hidden = h_n[-1]

        stat = self.relu(self.static_data(static_data))
        food_int = self.relu(self.food_intake(food_intake))

        # merge all
        x = torch.cat([hidden, stat, food_int], dim=1)
        x = self.relu(self.combined_layers(x))

        return self.dose_insulin(x), self.dose_drug_tabletes(x)
