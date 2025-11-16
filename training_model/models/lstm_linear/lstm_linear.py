import torch
import torch.nn as nn


class LSTMLinearModel(nn.Module):
    def __init__(self, static_dim, hidden_size, num_insulin_types, num_drug_types):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        # self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        self.static_data = nn.Linear(static_dim, 32)
        self.food_intake = nn.Linear(1, 32)

        self.combined_layers = nn.Linear(hidden_size + 32 + 32, 64)
        self.combined_bn = nn.BatchNorm1d(64)

        # for doses:
        self.dose_insulin = nn.Linear(64, num_insulin_types)
        self.dose_drug_tabletes = nn.Linear(64, num_drug_types)

        # for availability of drugs:
        self.insulin_mask = nn.Linear(64, num_insulin_types)
        self.drug_tablet_mask = nn.Linear(64, num_drug_types)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic_data, static_data, food_intake):
        # dynamic_data: (B, T, 3)
        _, (h_n, _) = self.lstm(dynamic_data)
        # _, h_n = self.gru(dynamic_data)

        # h_n shape: (num_layers, B, hidden_size)
        # hidden = h_n[-1]  # (B, hidden_size) → 2D
        hidden = self.lstm_norm(h_n[-1])

        stat = self.dropout(self.relu(self.static_data(static_data)))  # (B, 32)
        food_int = self.dropout(self.relu(self.food_intake(food_intake)))  # (B, 32)

        x = torch.cat([hidden, stat, food_int], dim=1)  # (B, 128)
        x = self.combined_bn(self.relu(self.combined_layers(x)))  # (B, 64)
        x = self.dropout(x)

        return (
            self.dose_insulin(x),
            self.dose_drug_tabletes(x),
            self.insulin_mask(x),
            self.drug_tablet_mask(x)
        )

    def extract_features(self, dynamic_data, static_data, food_intake):
        with torch.no_grad():
            _, (h_n, _) = self.lstm(dynamic_data)
            hidden = self.lstm_norm(h_n[-1])

            stat = self.relu(self.static_data(static_data))
            food_int = self.relu(self.food_intake(food_intake))

            x = torch.cat([hidden, stat, food_int], dim=1)
            x = self.combined_bn(self.relu(self.combined_layers(x)))
            x = self.dropout(x)
        return x
