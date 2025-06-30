import tensorflow as tf
import numpy as np

class DiabetesDataset:
    def __init__(self, df, seq_cols, static_cols, seq_len=20):
        self.df = df
        self.seq_cols = seq_cols
        self.static_cols = static_cols
        self.seq_len = seq_len
        self.patients = df['Patient Number'].unique()
        self.num_seq_features = len(seq_cols)

    def get_dataset(self):
        def gen():
            for pid in self.patients:
                patient_df = self.df[self.df['Patient Number'] == pid]
                static = patient_df.iloc[0][self.static_cols].values.astype(np.float32)
                seq = patient_df[self.seq_cols].tail(self.seq_len).values.astype(np.float32)
                seq = self._pad_sequence(seq)

                therapy = self._determine_therapy_type(patient_df)
                insulin = patient_df[[col for col in patient_df.columns if col.startswith('dose_') and 'insulin' in col]].sum(axis=1).iloc[-1]
                tablet = patient_df[[col for col in patient_df.columns if col.startswith('dose_') and 'insulin' not in col]].sum(axis=1).iloc[-1]

                yield (seq, static), (therapy, insulin, tablet)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (tf.TensorSpec(shape=(self.seq_len, self.num_seq_features), dtype=tf.float32),
                 tf.TensorSpec(shape=(len(self.static_cols),), dtype=tf.float32)),
                (tf.TensorSpec(shape=(), dtype=tf.int32),
                 tf.TensorSpec(shape=(), dtype=tf.float32),
                 tf.TensorSpec(shape=(), dtype=tf.float32))
            )
        )

    def _pad_sequence(self, seq):
        if len(seq) < self.seq_len:
            pad_len = self.seq_len - len(seq)
            pad = np.zeros((pad_len, self.num_seq_features), dtype=np.float32)
            seq = np.vstack((pad, seq))
        elif len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]
        return seq

    @staticmethod
    def _determine_therapy_type(df):
        last = df.iloc[-1]
        has_insulin = any(last[col] > 0 for col in df.columns if 'dose_' in col and 'insulin' in col)
        has_tablet = any(last[col] > 0 for col in df.columns if 'dose_' in col and 'insulin' not in col)
        if has_insulin and has_tablet:
            return 2
        elif has_insulin:
            return 1
        elif has_tablet:
            return 0
        return 0
