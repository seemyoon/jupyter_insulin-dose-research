# Insulin Dose Prediction

Multi-task deep learning system that predicts **which drugs** to prescribe (classification) and **what dose** for each (regression) for diabetes patients, using temporal glucose measurements and static patient features.

## Architecture

The system supports three model variants:

| Model | Description |
|-------|-------------|
| **GRU** | GRU encoder for time-series + static feature fusion |
| **LSTM** | LSTM encoder with separate cell state for longer memory |
| **Hybrid XGBoost + LSTM** | XGBoost learns dose baseline from tabular data; LSTM refines using temporal patterns |

All variants share a common multi-head output:
- **Dose regression heads** — predict normalized dose for each insulin/tablet slot
- **Drug selection heads** — predict whether each drug should be prescribed (binary per slot)

The loss combines `masked_mse` (regression on active drugs only) with `BCEWithLogitsLoss` using per-drug-type `pos_weight` to handle class imbalance.

Open `architecture.html` in a browser for a full visual diagram.

## Prerequisites

- Python 3.10+
- PostgreSQL (see [Set_up_PostgreSQL_on_Ubuntu.md](Set_up_PostgreSQL_on_Ubuntu.md))

## Setup

1. **Clone and install dependencies**

```bash
git clone <repo-url>
cd jupyter_insulin-dose-research
pip install -r requirements.txt
```

2. **Configure the database**

Create a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=diabetes
DB_USER=your_user
DB_PASSWORD=your_password
```

3. **Create tables and load data**

```bash
python db/create_tables.py
python load_data_controller.py
```

This imports `data/Shanghai_data.csv` into PostgreSQL.

## Usage

### Train a model (CLI)

```bash
python main.py
```

Edit the bottom of `main.py` to change settings:

```python
pipeline = FullModel()
pipeline.main(
    search_method="optuna",   # "optuna" or "grid"
    n_trials=30,              # Optuna trials
    model_type="lstm",        # "gru", "lstm", or "hybrid_xgb_lstm"
    loss_weight_bce=1.0,      # BCE/MSE balance (fixed, not tuned)
)
```

This runs hyperparameter search, evaluates on a held-out test set, and saves `checkpoint.pt`.

### Train a model (Streamlit UI)

```bash
streamlit run app.py
```

Features:
- Select model architecture (GRU / LSTM / Hybrid)
- Choose Optuna or Grid Search
- Configure epochs, patience, BCE weight
- Live progress table with classification + regression metrics
- Compare runs across experiments
- Saves `checkpoint.pt` automatically

### Predict for a patient

```bash
streamlit run predict_app.py
```

Loads `checkpoint.pt` (no database connection needed) and provides a form to enter:
- Patient demographics (age, gender, height, weight, ...)
- Medical data (diabetes type, HbA1c, lab values, ...)
- Additional drugs and comorbidities
- Recent CGM/CBG/blood ketone measurements
- Food intake and current therapy

Output shows:
- **Therapy type**: Insulin only / Tablets only / Combined
- **Drug prescriptions**: drug name, dose, confidence score
- **Full confidence overview**: all drug slots with probabilities

## Project Structure

```
├── main.py                    # FullModel pipeline: train, eval, checkpoint
├── app.py                     # Streamlit training UI
├── predict_app.py             # Streamlit inference UI
├── architecture.html          # Visual architecture diagram
├── requirements.txt           # Python dependencies
│
├── training_model/
│   ├── diabetes_model.py      # DiabetesModel (GRU/LSTM/Hybrid)
│   ├── glucose_dataset.py     # PyTorch Dataset + collate_fn
│   ├── repository.py          # SQLAlchemy queries
│   ├── xgboost_baseline.py    # XGBoost baseline for hybrid mode
│   └── preparing/
│       ├── static_preprocessing.py      # Feature scaling + embedding indices
│       └── static_embedding_encoder.py  # nn.Module for drug/comorb embeddings
│
├── utils/
│   ├── make_windows.py        # 24h sliding window builder
│   ├── split_dataset.py       # Patient-level train/test split
│   ├── to_datetime.py         # Time field → datetime
│   ├── convert_python_format.py
│   └── enums/
│       └── therapy_type.py    # TherapyType enum
│
├── db/
│   ├── engine.py              # PostgreSQL connection
│   ├── base.py                # SQLAlchemy Base
│   ├── create_tables.py       # Table creation script
│   └── models/                # 13 ORM models
│
├── preparing_data/            # CSV preprocessing (Shanghai dataset)
├── data/                      # Raw CSV files
├── alembic/                   # DB migrations
└── load_data_to_db.py         # CSV → PostgreSQL importer
```

## Metrics

The model is evaluated on two tasks:

### Drug Selection (Classification)
| Metric | Description |
|--------|-------------|
| Precision | Of predicted drugs, how many are actually prescribed? |
| Recall | Of actually prescribed drugs, how many does the model find? |
| F1 | Harmonic mean of precision and recall |
| Exact Match | Fraction of windows where all drug slots are correct |

### Dose Accuracy (Regression)
| Metric | Description |
|--------|-------------|
| R² | Coefficient of determination (1.0 = perfect; < 0 = worse than mean) |
| MAE | Mean absolute error in normalized dose units |

Both regression metrics are computed only on drug slots where the drug is actually prescribed (masked evaluation).

## Key Design Decisions

- **Patient-level splitting** — no patient appears in both train and test, preventing data leakage
- **pos_weight in BCE** — per-drug-type weighting (neg/pos ratio, capped at 15) to handle sparse prescriptions
- **Fixed loss_weight_bce** — the BCE/MSE balance is set before search, not tuned by Optuna, ensuring a consistent objective across trials
- **Scaler fitted on training data only** — `MinMaxScaler` and measurement normalization stats come from training patients exclusively
- **Checkpoint is self-contained** — `checkpoint.pt` includes model weights, architecture config, preprocessing artifacts, and drug name mappings; the inference app needs no database
