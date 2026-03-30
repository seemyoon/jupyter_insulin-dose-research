"""
Standalone inference app for Insulin / Tablet dose prediction.

Loads a checkpoint produced by the training pipeline and lets you
enter patient data to receive treatment recommendations:
  - Therapy type  (Insulin only / Tablets only / Combined)
  - Which specific drugs to prescribe
  - Dose for each prescribed drug
  - Confidence scores

Run from the project root:
    streamlit run predict_app.py
"""

from itertools import zip_longest

import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="Treatment Predictor", layout="wide")

# ═══════════════════════════════════════════════════════════════════
#  Load checkpoint
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from training_model.diabetes_model import DiabetesModel

    model = DiabetesModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


# ── sidebar ───────────────────────────────────────────────────────

st.sidebar.title("Model")
ckpt_path = st.sidebar.text_input("Checkpoint file", value="checkpoint.pt")

try:
    ckpt, model = _load_checkpoint(ckpt_path)
except FileNotFoundError:
    st.error(
        f"Checkpoint **`{ckpt_path}`** not found.  \n"
        "Train a model first (via `main.py` or `app.py`) to produce the file."
    )
    st.stop()
except Exception as exc:
    st.error(f"Failed to load checkpoint: {exc}")
    st.stop()

cfg = ckpt["model_config"]
preproc = ckpt["preprocessing"]
maps = ckpt["mappings"]

MODEL_LABELS = {"gru": "GRU", "lstm": "LSTM", "hybrid_xgb_lstm": "Hybrid XGBoost + LSTM"}
st.sidebar.success(f"**{MODEL_LABELS.get(cfg['model_type'], cfg['model_type'])}** loaded")
st.sidebar.caption(
    f"Insulin types: {cfg['num_insulin_types']}  \n"
    f"Tablet types: {cfg['num_drug_types']}"
)

THRESHOLD = st.sidebar.slider(
    "Decision threshold", 0.1, 0.9, 0.5, 0.05,
    help="Drugs with confidence above this threshold are prescribed.",
)

# ═══════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════

st.title("Treatment Prediction")
st.caption(
    "Enter patient data below and press **Predict** to receive "
    "drug and dose recommendations."
)

# ═══════════════════════════════════════════════════════════════════
#  Input form
# ═══════════════════════════════════════════════════════════════════

with st.form("patient_form"):

    # ── demographics ──────────────────────────────────────────────
    st.subheader("Demographics")
    d1, d2, d3 = st.columns(3)
    with d1:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Male", "Female"][x])
        age = st.number_input("Age (years)", 1, 120, 50)
    with d2:
        height = st.number_input("Height (cm)", 50.0, 250.0, 170.0, step=0.5)
        weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, step=0.5)
    with d3:
        smoking = st.selectbox(
            "Smoking history", [0, 1, 2],
            format_func=lambda x: ["No", "Former", "Current"][x],
        )
        alcohol = st.selectbox(
            "Alcohol history", [0, 1, 2],
            format_func=lambda x: ["No", "Former", "Current"][x],
        )

    # ── medical data ──────────────────────────────────────────────
    st.subheader("Medical Data")
    m1, m2 = st.columns(2)
    with m1:
        diabetes_type = st.selectbox("Diabetes type", [1, 2])
        duration = st.number_input("Duration (years)", 0.0, 80.0, 5.0, step=0.5)
        fasting_gluc = st.number_input("Fasting glucose (mmol/L)", 0.0, 35.0, 7.0, step=0.1)
        pp_gluc = st.number_input("Postprandial glucose (mmol/L)", 0.0, 45.0, 10.0, step=0.1)
        fasting_cpep = st.number_input("Fasting C-peptide (ng/mL)", 0.0, 20.0, 1.5, step=0.1)
        pp_cpep = st.number_input("Postprandial C-peptide (ng/mL)", 0.0, 30.0, 3.0, step=0.1)
        fasting_ins = st.number_input("Fasting insulin (uIU/mL)", 0.0, 100.0, 10.0, step=0.5)
        pp_ins = st.number_input("Postprandial insulin (uIU/mL)", 0.0, 300.0, 40.0, step=1.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 20.0, 7.5, step=0.1)
    with m2:
        glyc_alb = st.number_input("Glycated albumin (%)", 5.0, 60.0, 20.0, step=0.5)
        chol = st.number_input("Total cholesterol (mmol/L)", 1.0, 15.0, 5.0, step=0.1)
        trig = st.number_input("Triglycerides (mmol/L)", 0.1, 20.0, 1.5, step=0.1)
        hdl = st.number_input("HDL (mmol/L)", 0.1, 5.0, 1.2, step=0.1)
        ldl = st.number_input("LDL (mmol/L)", 0.1, 10.0, 3.0, step=0.1)
        creatinine = st.number_input("Creatinine (umol/L)", 10.0, 1500.0, 80.0, step=1.0)
        egfr = st.number_input("eGFR (mL/min/1.73m2)", 3.0, 150.0, 90.0, step=1.0)
        uric_acid = st.number_input("Uric acid (umol/L)", 30.0, 900.0, 350.0, step=5.0)
        bun = st.number_input("BUN (mmol/L)", 0.5, 40.0, 5.0, step=0.1)

    # ── additional drugs & comorbidities ──────────────────────────
    st.subheader("Additional Drugs & Comorbidities")
    ad1, ad2 = st.columns(2)
    drug_name_map = maps.get("drug_names", {})
    comorb_name_map = maps.get("comorbidity_names", {})
    with ad1:
        selected_drugs = st.multiselect(
            "Additional drugs the patient takes",
            options=sorted(drug_name_map.keys()),
            format_func=lambda x: drug_name_map.get(x, f"Drug #{x}"),
        )
    with ad2:
        selected_comorbs = st.multiselect(
            "Comorbidities",
            options=sorted(comorb_name_map.keys()),
            format_func=lambda x: comorb_name_map.get(x, f"#{x}"),
        )

    # ── measurements (24-hour window) ─────────────────────────────
    st.subheader("Recent Measurements (24 h)")
    st.caption("Enter comma-separated values. Leave blank if not available.")
    me1, me2, me3 = st.columns(3)
    with me1:
        cgm_raw = st.text_area("CGM values", placeholder="5.5, 6.2, 7.1, 8.0")
    with me2:
        cbg_raw = st.text_area("CBG values", placeholder="6.0, 7.5")
    with me3:
        ketone_raw = st.text_area("Blood ketone values", placeholder="0.3, 0.2")

    # ── other ─────────────────────────────────────────────────────
    st.subheader("Context")
    o1, o2 = st.columns(2)
    with o1:
        food_count = st.number_input("Meals in past 24 h", 0, 20, 3)
    with o2:
        therapy_option = st.selectbox(
            "Current therapy (if known)",
            ["Unknown", "Insulin only", "Combined", "Tablet only"],
            help="What the patient is currently prescribed. "
                 "Select 'Unknown' for a new patient.",
        )

    submitted = st.form_submit_button("Predict Treatment", type="primary")


# ═══════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════

def _parse_csv(text: str):
    if not text or not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


if submitted:
    # ── build static feature vector (same order as training) ──────
    features = [
        gender, age, height, weight, smoking, alcohol,
        diabetes_type, duration, fasting_gluc, pp_gluc,
        fasting_cpep, pp_cpep, fasting_ins, pp_ins,
        hba1c, glyc_alb, chol, trig, hdl, ldl,
        creatinine, egfr, uric_acid, bun,
    ]
    static_norm = preproc["scaler"].transform([features])
    static_t = torch.tensor(static_norm, dtype=torch.float32)         # (1, F)

    # ── drug / comorbidity embedding indices ──────────────────────
    unique_drugs = maps["unique_drugs"]
    drug_idx = [unique_drugs[d] for d in selected_drugs if d in unique_drugs] or [0]
    drug_t = torch.tensor([drug_idx], dtype=torch.long)                # (1, D)

    unique_comorbs = maps["unique_comorbities"]
    comorb_idx = [unique_comorbs[c] for c in selected_comorbs if c in unique_comorbs] or [0]
    comorb_t = torch.tensor([comorb_idx], dtype=torch.long)            # (1, C)

    # ── measurements ──────────────────────────────────────────────
    cgm = _parse_csv(cgm_raw)
    cbg = _parse_csv(cbg_raw)
    ketones = _parse_csv(ketone_raw)

    if not cgm and not cbg and not ketones:
        cgm = [0.0]

    rows = [[c, b, k] for c, b, k in zip_longest(cgm, cbg, ketones, fillvalue=0.0)]
    meas_t = torch.tensor([rows], dtype=torch.float32)                 # (1, T, 3)
    meas_mean = torch.tensor(preproc["meas_mean"], dtype=torch.float32)
    meas_std = torch.tensor(preproc["meas_std"], dtype=torch.float32).clamp(min=1e-8)
    meas_t = (meas_t - meas_mean) / meas_std

    # ── auxiliary inputs ──────────────────────────────────────────
    food_t = torch.tensor([[food_count]], dtype=torch.float32)         # (1, 1)

    therapy_map = {"Unknown": None, "Insulin only": 0, "Combined": 1, "Tablet only": 2}
    therapy_val = therapy_map[therapy_option]
    therapy_t = torch.zeros(1, 3, dtype=torch.float32)
    if therapy_val is not None:
        therapy_t[0, therapy_val] = 1.0

    # ── XGBoost baseline (hybrid only) ────────────────────────────
    xgb_t = None
    if cfg["model_type"] == "hybrid_xgb_lstm" and ckpt.get("xgb_model") is not None:
        xgb_obj = ckpt["xgb_model"]
        xgb_input = np.array(static_norm, dtype=np.float32)
        xgb_pred = xgb_obj.model.predict(xgb_input)
        xgb_t = torch.tensor(xgb_pred, dtype=torch.float32)           # (1, out_dim)

    # ── forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        dose_ins, dose_tab, logit_ins, logit_tab = model(
            meas_t, static_t, drug_t, comorb_t,
            food_t, therapy_t, xgb_preds=xgb_t,
        )

    prob_ins = torch.sigmoid(logit_ins).squeeze(0).numpy()
    prob_tab = torch.sigmoid(logit_tab).squeeze(0).numpy()
    dose_ins_vals = dose_ins.squeeze(0).numpy() * preproc["max_insulin"]
    dose_tab_vals = dose_tab.squeeze(0).numpy() * preproc["max_tablet"]

    sel_ins = prob_ins >= THRESHOLD
    sel_tab = prob_tab >= THRESHOLD

    has_insulin = sel_ins.any()
    has_tablets = sel_tab.any()

    # ── determine therapy type ────────────────────────────────────
    if has_insulin and has_tablets:
        therapy_label = "Combined (Insulin + Tablets)"
    elif has_insulin:
        therapy_label = "Insulin Only"
    elif has_tablets:
        therapy_label = "Tablets Only"
    else:
        therapy_label = "No Active Treatment Predicted"

    # ═════════════════════════════════════════════════════════════
    #  Results
    # ═════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("Treatment Recommendation")
    st.info(f"**Therapy type:** {therapy_label}")

    insulin_names = maps.get("insulin_names", {})
    tablet_names = maps.get("tablet_names", {})

    col_left, col_right = st.columns(2)

    # ── insulin table ─────────────────────────────────────────────
    with col_left:
        st.markdown("**Insulin**")
        ins_rows = []
        for i in range(len(prob_ins)):
            if sel_ins[i]:
                ins_rows.append({
                    "Drug": insulin_names.get(i, f"Insulin #{i}"),
                    "Dose (units)": round(float(max(dose_ins_vals[i], 0)), 1),
                    "Confidence": f"{prob_ins[i]:.1%}",
                })
        if ins_rows:
            st.dataframe(
                pd.DataFrame(ins_rows),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No insulin predicted above threshold.")

    # ── tablet table ──────────────────────────────────────────────
    with col_right:
        st.markdown("**Tablets**")
        tab_rows = []
        for i in range(len(prob_tab)):
            if sel_tab[i]:
                tab_rows.append({
                    "Drug": tablet_names.get(i, f"Tablet #{i}"),
                    "Dose (mg)": round(float(max(dose_tab_vals[i], 0)), 2),
                    "Confidence": f"{prob_tab[i]:.1%}",
                })
        if tab_rows:
            st.dataframe(
                pd.DataFrame(tab_rows),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No tablets predicted above threshold.")

    # ── full confidence overview ──────────────────────────────────
    with st.expander("Full confidence scores (all drug slots)", expanded=False):
        all_ins = []
        for i in range(len(prob_ins)):
            name = insulin_names.get(i, f"Insulin #{i}")
            all_ins.append({
                "Slot": i,
                "Drug": name,
                "Confidence": round(float(prob_ins[i]), 4),
                "Dose (raw)": round(float(dose_ins_vals[i]), 3),
                "Selected": bool(sel_ins[i]),
            })
        all_tab = []
        for i in range(len(prob_tab)):
            name = tablet_names.get(i, f"Tablet #{i}")
            all_tab.append({
                "Slot": i,
                "Drug": name,
                "Confidence": round(float(prob_tab[i]), 4),
                "Dose (raw)": round(float(dose_tab_vals[i]), 3),
                "Selected": bool(sel_tab[i]),
            })

        st.markdown("*Insulin slots*")
        st.dataframe(pd.DataFrame(all_ins), use_container_width=True, hide_index=True)
        st.markdown("*Tablet slots*")
        st.dataframe(pd.DataFrame(all_tab), use_container_width=True, hide_index=True)
