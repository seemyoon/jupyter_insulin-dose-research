import torch
import torch.nn as nn

from training_model.preparing.static_embedding_encoder import StaticEmbedderEncoder

NUM_THERAPY_TYPES = 3
VALID_MODEL_TYPES = ("gru", "lstm", "hybrid_xgb_lstm")


class DiabetesModel(nn.Module):
    """
    Switchable architecture for insulin / tablet dose prediction.

    model_type
    ----------
    "gru"             – GRU encoder for time-series.
    "lstm"            – LSTM encoder for time-series.
    "hybrid_xgb_lstm" – LSTM encoder with XGBoost dose-baseline predictions
                        injected as additional features into the combined layer.
    """

    def __init__(self, static_dim, hidden_size, num_insulin_types, num_drug_types,
                 unique_drugs_size, unique_comorbities_size, emb_dim=32,
                 num_rnn_layers=1, dropout=0.2,
                 model_type="gru", xgb_feature_dim=0):
        super().__init__()

        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'"
            )

        self.model_type = model_type

        # ── static branch (trainable embeddings + MLP) ──────────────
        static_hidden = max(hidden_size // 2, 16)
        self.static_encoder = StaticEmbedderEncoder(
            static_dim=static_dim,
            unique_drugs_size=unique_drugs_size,
            unique_comorbities_size=unique_comorbities_size,
            emb_dim=emb_dim,
            hidden_dim=static_hidden,
        )

        # ── recurrent branch ────────────────────────────────────────
        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        # ── auxiliary inputs ────────────────────────────────────────
        aux_dim = 16
        self.food_intake_fc = nn.Linear(1, aux_dim)
        self.therapy_fc = nn.Linear(NUM_THERAPY_TYPES, aux_dim)

        # ── combined → heads ────────────────────────────────────────
        combined_dim = hidden_size + static_hidden + aux_dim * 2
        if model_type == "hybrid_xgb_lstm":
            combined_dim += xgb_feature_dim

        head_dim = hidden_size
        self.combined_fc = nn.Linear(combined_dim, head_dim)

        self.dose_insulin = nn.Linear(head_dim, num_insulin_types)
        self.dose_drug_tablets = nn.Linear(head_dim, num_drug_types)
        self.insulin_mask = nn.Linear(head_dim, num_insulin_types)
        self.drug_tablet_mask = nn.Linear(head_dim, num_drug_types)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # -----------------------------------------------------------------

    def forward(self, dynamic_data, static_tensor, drug_indices, comorb_indices,
                food_intake, therapy_type, xgb_preds=None):
        """
        :param dynamic_data:   (B, T, 3)  z-score-normalized measurements
        :param static_tensor:  (B, static_dim)
        :param drug_indices:   (B, max_drugs) LongTensor
        :param comorb_indices: (B, max_comorbs) LongTensor
        :param food_intake:    (B, 1)
        :param therapy_type:   (B, 3) one-hot
        :param xgb_preds:      (B, xgb_feature_dim) – only for hybrid_xgb_lstm
        """
        # ── RNN hidden state ────────────────────────────────────────
        if self.model_type == "gru":
            _, h_n = self.rnn(dynamic_data)
        else:  # LSTM or hybrid
            _, (h_n, _) = self.rnn(dynamic_data)
        hidden = h_n[-1]  # (B, hidden_size)

        # ── static features ─────────────────────────────────────────
        stat = self.static_encoder(static_tensor, drug_indices, comorb_indices)

        # ── auxiliary ────────────────────────────────────────────────
        food = self.dropout(self.relu(self.food_intake_fc(food_intake)))
        therapy = self.dropout(self.relu(self.therapy_fc(therapy_type)))

        # ── combine ─────────────────────────────────────────────────
        parts = [hidden, stat, food, therapy]
        if self.model_type == "hybrid_xgb_lstm" and xgb_preds is not None:
            parts.append(xgb_preds)

        x = torch.cat(parts, dim=1)
        x = self.dropout(self.relu(self.combined_fc(x)))

        return (
            self.dose_insulin(x),
            self.dose_drug_tablets(x),
            self.insulin_mask(x),
            self.drug_tablet_mask(x),
        )
