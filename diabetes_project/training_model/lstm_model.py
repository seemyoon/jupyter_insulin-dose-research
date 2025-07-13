import tensorflow as tf
from tensorflow.keras import layers, Model


class DiabetesLSTMModel(Model):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True)
        self.static_fc = layers.Dense(hidden_dim, activation='relu')
        self.combined_fc = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)

        self.therapy_head = layers.Dense(3)
        self.insulin_head = layers.Dense(1)
        self.tablet_head = layers.Dense(1)

    def call(self, inputs, training=False):
        seq_input, static_input = inputs
        lstm_out = self.lstm(seq_input)
        lstm_last = lstm_out[:, -1, :]
        static_feat = self.static_fc(static_input)

        combined = tf.concat([lstm_last, static_feat], axis=1)
        combined = self.combined_fc(combined)
        if training:
            combined = self.dropout(combined, training=training)

        therapy_logits = self.therapy_head(combined)
        insulin_dose = self.insulin_head(combined)
        tablet_dose = self.tablet_head(combined)

        return therapy_logits, insulin_dose, tablet_dose
