import torch
import torch.nn as nn


class DiabetesModel(nn.Module):
    def __init__(self, static_dim, hidden_size, num_insulin_types, num_drug_types):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        # self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)

        self.static_data = nn.Linear(static_dim, 32)
        self.food_intake = nn.Linear(1, 32)

        self.combined_layers = nn.Linear(hidden_size + 32 + 32, 64)

        self.dose_insulin = nn.Linear(64, num_insulin_types)

        self.dose_drug_tabletes = nn.Linear(64, num_drug_types)

        self.relu = nn.ReLU()

    def forward(self, dynamic_data, static_data, food_intake):
        # dynamic_data: (B, T, 3)
        _, (h_n, _) = self.lstm(dynamic_data)
        # _, h_n = self.gru(dynamic_data)

        # h_n shape: (num_layers, B, hidden_size)
        hidden = h_n[-1]  # (B, hidden_size) → 2D

        stat = self.relu(self.static_data(static_data))  # (B, 32)
        food_int = self.relu(self.food_intake(food_intake))  # (B, 32)

        x = torch.cat([hidden, stat, food_int], dim=1)  # (B, 128)
        x = self.relu(self.combined_layers(x))  # (B, 64)

        return self.dose_insulin(x), self.dose_drug_tabletes(x)

