import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import tensorflow as tf

from diabetes_project.third_step.training_model.dataset import DiabetesDataset
from diabetes_project.third_step.training_model.lstm_model import DiabetesLSTMModel

loss_fn_therapy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn_mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


@tf.function
def train_step(model, inputs, targets):
    (seq_input, static_input) = inputs
    (therapy_target, insulin_target, tablet_target) = targets

    with tf.GradientTape() as tape:
        therapy_logits, insulin_pred, tablet_pred = model((seq_input, static_input), training=True)

        loss_therapy = loss_fn_therapy(therapy_target, therapy_logits)
        loss_insulin = loss_fn_mse(insulin_target, tf.squeeze(insulin_pred, axis=1))
        loss_tablet = loss_fn_mse(tablet_target, tf.squeeze(tablet_pred, axis=1))

        loss = loss_therapy + 0.5 * loss_insulin + 0.5 * loss_tablet

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


seq_cols = [
    'CGM (mg / dl)', 'CBG (mg / dl)', 'Blood Ketone (mmol / L)',
    'Fasting Plasma Glucose (mg/dl)', '2-hour Postprandial Plasma Glucose (mg/dl)',
    'HbA1c (mmol/mol)', 'Glycated Albumin (%)', 'Total Cholesterol (mmol/L)',
    'Triglyceride (mmol/L)', 'Estimated Glomerular Filtration Rate  (ml/min/1.73m2)'
]

static_cols = [
    'Gender (Female=1, Male=2)', 'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
    'Smoking History (pack year)'
]

BATCH_SIZE = 32
EPOCHS = 10
SEQ_LEN = 20
INPUT_DIM = len(seq_cols)
STATIC_DIM = len(static_cols)
HIDDEN_DIM = 64

df = pd.read_csv(Path('../data/train_data.csv'))

dataset = DiabetesDataset(df, seq_cols, static_cols, seq_len=SEQ_LEN).get_dataset()
dataset = dataset.shuffle(100).batch(BATCH_SIZE)

model = DiabetesLSTMModel(INPUT_DIM, HIDDEN_DIM, STATIC_DIM)

for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0

    for inputs, targets in dataset:
        loss = train_step(model, inputs, targets)
        total_loss += loss.numpy()
        steps += 1

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / steps:.4f}")
