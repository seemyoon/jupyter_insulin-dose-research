"""
Streamlit UI for the Insulin Dose Prediction pipeline.

Run from the project root:
    streamlit run app.py
"""

import time
from datetime import datetime
from itertools import product

import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

# ── page config (must be first Streamlit call) ──────────────────────
st.set_page_config(page_title="Insulin Dose Prediction", layout="wide")

# ── lazy project imports (DB connects on import) ────────────────────
_IMPORT_ERROR = None

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from main import (
        FullModel,
        set_seed,
        compute_measurement_stats,
        compute_max_doses,
        split_by_patient,
        _forward_batch,
        masked_mse,
    )
    from training_model.diabetes_model import VALID_MODEL_TYPES
    from training_model.glucose_dataset import GlucoseDataset
except Exception as exc:
    _IMPORT_ERROR = str(exc)
    VALID_MODEL_TYPES = ("gru", "lstm", "hybrid_xgb_lstm")

# ── session state defaults ──────────────────────────────────────────
_DEFAULTS = {
    "pipeline": None,
    "data_loaded": False,
    "data_summary": None,
    "run_history": [],
    "best_run": None,
    "best_model": None,
    "best_test_ds": None,
    "training_active": False,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

MODEL_LABELS = {
    "gru": "GRU",
    "lstm": "LSTM",
    "hybrid_xgb_lstm": "Hybrid XGBoost + LSTM",
}


# ════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════

st.sidebar.title("Configuration")

model_type = st.sidebar.selectbox(
    "Model architecture",
    options=list(VALID_MODEL_TYPES),
    format_func=lambda k: MODEL_LABELS.get(k, k),
)

search_method = st.sidebar.selectbox(
    "Hyperparameter search",
    options=["optuna", "grid"],
    format_func=lambda k: {"optuna": "Optuna (Bayesian)", "grid": "Grid Search"}[k],
)

st.sidebar.divider()
st.sidebar.subheader("Training controls")
n_trials = st.sidebar.slider(
    "Trials" if search_method == "optuna" else "Combinations (auto)",
    min_value=3, max_value=100, value=20,
    disabled=(search_method == "grid"),
)
epochs = st.sidebar.slider("Max epochs per trial", 10, 150, 30)
patience = st.sidebar.slider("Early stopping patience", 2, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Loss balance")
loss_weight_bce = st.sidebar.slider(
    "BCE weight (classification vs regression)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Fixed across all trials. Higher → more weight on drug selection; "
         "lower → more weight on dose accuracy.",
)

with st.sidebar.expander("Search space details"):
    if search_method == "optuna":
        st.markdown(
            "| Param | Range |\n"
            "|---|---|\n"
            "| hidden_size | 32 .. 256 (log) |\n"
            "| num_rnn_layers | 1 .. 3 |\n"
            "| dropout | 0.05 .. 0.5 |\n"
            "| lr | 1e-5 .. 1e-2 (log) |\n"
            "| batch_size | 16, 32, 64 |\n"
            "| emb_dim | 16, 32, 64 |"
        )
    else:
        st.markdown(
            "| Param | Values |\n"
            "|---|---|\n"
            "| hidden_size | 64, 128 |\n"
            "| dropout | 0.1, 0.2, 0.3 |\n"
            "| lr | 1e-3, 3e-4 |\n"
            "| batch_size | 16, 32 |"
        )


# ════════════════════════════════════════════════════════════════════
#  Header
# ════════════════════════════════════════════════════════════════════

st.title("Insulin Dose Prediction")
st.caption("Train, evaluate, and compare GRU / LSTM / Hybrid XGBoost+LSTM models")

if _IMPORT_ERROR:
    st.error(
        f"**Failed to load project modules.** Make sure PostgreSQL is running "
        f"and `.env` is configured.\n\n`{_IMPORT_ERROR}`"
    )
    st.stop()

tab_train, tab_results, tab_data = st.tabs(["Training", "Results", "Data overview"])


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _load_pipeline():
    """Create FullModel and load data (cached in session state)."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = FullModel()
    pipeline = st.session_state.pipeline
    pipeline._load_data_once()
    data = pipeline._data_cache

    n_patients = len(data["patient_ids"])
    n_windows = len(data["windows"])
    n_insulin = data["num_insulin"]
    n_tablets = data["num_drug_types"]
    st.session_state.data_summary = {
        "patients": n_patients,
        "windows": n_windows,
        "insulin_types": n_insulin,
        "tablet_types": n_tablets,
    }
    st.session_state.data_loaded = True
    return pipeline


def _run_grid(pipeline, model_type, epochs, patience, bce_weight, progress_bar, table_slot):
    grid = {
        "hidden_size": [64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-3, 3e-4],
        "batch_size": [16, 32],
    }
    combos = list(product(*grid.values()))
    total = len(combos)
    rows = []

    best_loss = float("inf")
    best_model = None
    best_test_ds = None
    best_config = {}
    best_artifacts = {}

    for i, (hs, do, lr, bs) in enumerate(combos):
        metrics, model, test_ds = pipeline.train_and_eval(
            hs, do, lr, bs,
            epochs=epochs, patience=patience, model_type=model_type,
            loss_weight_bce=bce_weight,
        )
        row = {
            "hidden": hs, "dropout": do, "lr": lr, "batch": bs,
            "val_loss": round(metrics["val_loss"], 5),
            "F1_ins": round(metrics["f1_insulin"], 3),
            "F1_tab": round(metrics["f1_tablets"], 3),
            "ExM_ins": round(metrics["exact_match_insulin"], 3),
            "R2_ins": round(metrics["r2_insulin"], 4),
            "MAE_ins": round(metrics["mae_insulin"], 4),
        }
        rows.append(row)
        table_slot.dataframe(pd.DataFrame(rows), width='stretch')
        progress_bar.progress((i + 1) / total, text=f"Trial {i+1}/{total}")

        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_model = model
            best_test_ds = test_ds
            best_config = dict(hidden_size=hs, dropout=do, lr=lr,
                               batch_size=bs, loss_weight_bce=bce_weight)
            best_artifacts = dict(pipeline._last_train_artifacts)

    pipeline._last_train_artifacts = best_artifacts
    return rows, best_model, best_test_ds, best_config, best_loss


def _run_optuna(pipeline, model_type, n_trials, epochs, patience, bce_weight,
                progress_bar, table_slot):
    rows = []
    holder = {"model": None, "test_ds": None, "loss": float("inf"),
              "config": {}, "artifacts": {}}

    def objective(trial):
        hs = trial.suggest_int("hidden_size", 32, 256, log=True)
        nl = trial.suggest_int("num_rnn_layers", 1, 3)
        do = trial.suggest_float("dropout", 0.05, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [16, 32, 64])
        ed = trial.suggest_categorical("emb_dim", [16, 32, 64])

        metrics, model, test_ds = pipeline.train_and_eval(
            hs, do, lr, bs,
            emb_dim=ed, num_rnn_layers=nl,
            epochs=epochs, patience=patience,
            model_type=model_type,
            loss_weight_bce=bce_weight,
        )

        row = {
            "trial": trial.number + 1,
            "hidden": hs, "layers": nl, "dropout": round(do, 3),
            "lr": f"{lr:.2e}", "batch": bs, "emb": ed,
            "val_loss": round(metrics["val_loss"], 5),
            "F1_ins": round(metrics["f1_insulin"], 3),
            "F1_tab": round(metrics["f1_tablets"], 3),
            "ExM_ins": round(metrics["exact_match_insulin"], 3),
            "R2_ins": round(metrics["r2_insulin"], 4),
        }
        rows.append(row)
        table_slot.dataframe(pd.DataFrame(rows), use_container_width=True)
        progress_bar.progress(
            (trial.number + 1) / n_trials,
            text=f"Trial {trial.number + 1}/{n_trials}",
        )

        if metrics["val_loss"] < holder["loss"]:
            holder.update(
                model=model, test_ds=test_ds, loss=metrics["val_loss"],
                config=dict(hidden_size=hs, num_rnn_layers=nl, dropout=do,
                            lr=lr, batch_size=bs, emb_dim=ed),
                artifacts=dict(pipeline._last_train_artifacts),
            )
        return metrics["val_loss"]

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    pipeline._last_train_artifacts = holder["artifacts"]
    return rows, holder["model"], holder["test_ds"], holder["config"], holder["loss"]


def _test_evaluation(model, test_ds):
    if model is None or test_ds is None or len(test_ds) == 0:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=GlucoseDataset.collate_fn)
    return FullModel._eval_metrics(model, loader, device)


# ════════════════════════════════════════════════════════════════════
#  Tab 1 — Training
# ════════════════════════════════════════════════════════════════════

with tab_train:
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.info(f"**Architecture:** {MODEL_LABELS[model_type]}")
    col_info2.info(f"**Search:** {search_method}  |  Epochs: {epochs}  |  Patience: {patience}")
    col_info3.info(
        f"**Trials:** {n_trials}" if search_method == "optuna"
        else f"**Combinations:** {len(list(product([64,128],[.1,.2,.3],[1e-3,3e-4],[16,32])))}"
    )

    st.divider()

    if st.button("Start training", type="primary", disabled=st.session_state.training_active):
        st.session_state.training_active = True

        with st.status("Loading data from database ...", expanded=True) as status:
            t0 = time.time()
            try:
                pipeline = _load_pipeline()
            except Exception as exc:
                st.error(f"Data loading failed: {exc}")
                st.session_state.training_active = False
                st.stop()
            load_sec = time.time() - t0
            status.update(label=f"Data loaded in {load_sec:.1f}s", state="complete")

        summ = st.session_state.data_summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", summ["patients"])
        c2.metric("Windows", summ["windows"])
        c3.metric("Insulin types", summ["insulin_types"])
        c4.metric("Tablet types", summ["tablet_types"])

        st.subheader("Hyperparameter search")
        progress = st.progress(0, text="Starting ...")
        results_table = st.empty()

        t0 = time.time()
        if search_method == "grid":
            rows, best_model, best_test_ds, best_cfg, best_loss = _run_grid(
                pipeline, model_type, epochs, patience, loss_weight_bce,
                progress, results_table,
            )
        else:
            rows, best_model, best_test_ds, best_cfg, best_loss = _run_optuna(
                pipeline, model_type, n_trials, epochs, patience,
                loss_weight_bce, progress, results_table,
            )
        elapsed = time.time() - t0

        progress.progress(1.0, text="Search complete")
        st.success(f"Finished in {elapsed/60:.1f} min.  Best val loss: **{best_loss:.5f}**")

        st.subheader("Best configuration")
        st.json(best_cfg)

        # Test-set evaluation
        st.subheader("Test-set evaluation")
        test_metrics = _test_evaluation(best_model, best_test_ds)
        if test_metrics:
            st.metric("Combined Loss", f"{test_metrics['val_loss']:.5f}")

            st.markdown("**Drug Selection (Classification)**")
            cls1, cls2 = st.columns(2)
            with cls1:
                st.markdown("*Insulin*")
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("Precision", f"{test_metrics['prec_insulin']:.3f}")
                ic2.metric("Recall", f"{test_metrics['recall_insulin']:.3f}")
                ic3.metric("F1", f"{test_metrics['f1_insulin']:.3f}")
                ic4.metric("Exact Match", f"{test_metrics['exact_match_insulin']:.3f}")
            with cls2:
                st.markdown("*Tablets*")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Precision", f"{test_metrics['prec_tablets']:.3f}")
                tc2.metric("Recall", f"{test_metrics['recall_tablets']:.3f}")
                tc3.metric("F1", f"{test_metrics['f1_tablets']:.3f}")
                tc4.metric("Exact Match", f"{test_metrics['exact_match_tablets']:.3f}")

            st.markdown("**Dose Accuracy (Regression)**")
            reg1, reg2 = st.columns(2)
            with reg1:
                st.markdown("*Insulin*")
                ri1, ri2 = st.columns(2)
                ri1.metric("R\u00b2", f"{test_metrics['r2_insulin']:.4f}")
                ri2.metric("MAE", f"{test_metrics['mae_insulin']:.4f}")
            with reg2:
                st.markdown("*Tablets*")
                rt1, rt2 = st.columns(2)
                rt1.metric("R\u00b2", f"{test_metrics['r2_tablets']:.4f}")
                rt2.metric("MAE", f"{test_metrics['mae_tablets']:.4f}")
        else:
            st.warning("Not enough test data to evaluate.")

        # Save full checkpoint (model + preprocessing + drug names)
        ckpt_path = "checkpoint.pt"
        pipeline.save_checkpoint(best_model, ckpt_path)
        st.info(f"Full checkpoint saved to `{ckpt_path}` — use with `predict_app.py`")

        # Persist run
        run_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model_type": model_type,
            "search": search_method,
            "best_config": best_cfg,
            "best_val_loss": best_loss,
            "test_metrics": test_metrics,
            "trials": rows,
            "elapsed_min": round(elapsed / 60, 2),
        }
        st.session_state.run_history.append(run_record)
        st.session_state.best_run = run_record
        st.session_state.best_model = best_model
        st.session_state.best_test_ds = best_test_ds
        st.session_state.training_active = False


# ════════════════════════════════════════════════════════════════════
#  Tab 2 — Results
# ════════════════════════════════════════════════════════════════════

with tab_results:
    history = st.session_state.run_history

    if not history:
        st.info("No training runs yet. Go to the **Training** tab to start.")
    else:
        st.subheader("Run history")

        summary_rows = []
        for i, run in enumerate(history):
            tm = run.get("test_metrics") or {}
            summary_rows.append({
                "#": i + 1,
                "Time": run["timestamp"],
                "Model": MODEL_LABELS.get(run["model_type"], run["model_type"]),
                "Search": run["search"],
                "Val Loss": round(run["best_val_loss"], 5),
                "Test Loss": round(tm.get("val_loss", 0), 5) if tm else "-",
                "F1 Ins": round(tm.get("f1_insulin", 0), 3) if tm else "-",
                "F1 Tab": round(tm.get("f1_tablets", 0), 3) if tm else "-",
                "ExM Ins": round(tm.get("exact_match_insulin", 0), 3) if tm else "-",
                "R2 Ins": round(tm.get("r2_insulin", 0), 4) if tm else "-",
                "MAE Ins": round(tm.get("mae_insulin", 0), 4) if tm else "-",
                "Time (min)": run["elapsed_min"],
            })
        st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)

        # Comparison chart
        if len(history) >= 2:
            st.subheader("Validation loss comparison")
            chart_df = pd.DataFrame({
                "Run": [f"#{i+1} {MODEL_LABELS.get(r['model_type'], r['model_type'])}" for i, r in enumerate(history)],
                "Val Loss": [r["best_val_loss"] for r in history],
            }).set_index("Run")
            st.bar_chart(chart_df)

        # Detailed view of the last run
        st.divider()
        st.subheader("Latest run details")
        latest = history[-1]
        st.write(f"**Model:** {MODEL_LABELS.get(latest['model_type'], latest['model_type'])}  |  "
                 f"**Search:** {latest['search']}  |  **Duration:** {latest['elapsed_min']} min")
        st.json(latest["best_config"])

        if latest["trials"]:
            with st.expander("All trial results", expanded=False):
                st.dataframe(pd.DataFrame(latest["trials"]), width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════
#  Tab 3 — Data overview
# ════════════════════════════════════════════════════════════════════

with tab_data:
    if not st.session_state.data_loaded:
        st.info("Data is loaded automatically when training starts. Press **Load data now** to preview statistics.")
        if st.button("Load data now"):
            with st.spinner("Connecting to database ..."):
                try:
                    _load_pipeline()
                except Exception as exc:
                    st.error(f"Could not load data: {exc}")
                    st.stop()
            st.rerun()
    else:
        summ = st.session_state.data_summary
        st.subheader("Dataset summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients (non-hospitalized)", summ["patients"])
        c2.metric("24-hour windows", summ["windows"])
        c3.metric("Insulin types", summ["insulin_types"])
        c4.metric("Tablet types", summ["tablet_types"])

        data = st.session_state.pipeline._data_cache
        windows = data["windows"]

        st.divider()
        st.subheader("Windows per patient")
        pid_counts = pd.Series([w["patient_id"] for w in windows]).value_counts()
        st.bar_chart(pid_counts.sort_index(), x_label="Patient ID", y_label="Number of windows")

        st.divider()
        st.subheader("Measurement density (CGM readings per window)")
        cgm_lens = [len(w["cgm_values"]) for w in windows]
        cgm_df = pd.DataFrame({"CGM readings per window": cgm_lens})
        st.bar_chart(cgm_df.value_counts().sort_index(), x_label="Readings", y_label="Windows")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Therapy type distribution")
            therapy_map = {0: "Insulin only", 1: "Combined", 2: "Tablet only", None: "None"}
            therapy_counts = pd.Series(
                [therapy_map.get(w.get("therapy_type"), "Unknown") for w in windows]
            ).value_counts()
            st.dataframe(therapy_counts.rename("Count"), width='stretch')

        with col_b:
            st.subheader("Static features (sample)")
            raw = data["raw_features"]
            cols = [
                "gender", "age", "height", "weight", "smoking", "alcohol",
                "diabetes_type", "duration", "fasting_gluc", "pp_gluc",
                "fasting_cpep", "pp_cpep", "fasting_ins", "pp_ins",
                "hba1c", "glyc_alb", "chol", "trig", "hdl", "ldl",
                "creatinine", "egfr", "uric_acid", "bun",
            ]
            sample = pd.DataFrame(raw[:10], columns=cols[:len(raw[0])])
            st.dataframe(sample, width='stretch')
