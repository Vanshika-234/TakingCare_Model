"""CareCompass dashboard.   Run with:  streamlit run dashboard.py

Pages
  1. Overview          - model performance, calibration, fairness audit, generative-model check
  2. Triage list       - every discharged patient ranked by predicted 30-day readmission risk
  3. Patient workspace - risk, drivers, similar patients, data-quality check, and the AI agent
  4. Review queue      - cases the agent escalated to a human clinician
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from config import ARTIFACTS, REPORTS, TIER_HIGH, TIER_MED
from src.agent import DischargeAgent
from src.system import FEATURES, CareSystem, tier

st.set_page_config(page_title="CareCompass", page_icon="🩺", layout="wide")
TIER_COLOR = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models...")
def load_agent() -> DischargeAgent:
    return DischargeAgent(CareSystem())


@st.cache_data(show_spinner="Scoring all patients...")
def score_cohort() -> pd.DataFrame:
    s = load_agent().sys
    p = s._ensemble_fn(s.pre.transform(s.test_df[FEATURES]))
    d = s.test_df[["age_num", "gender", "diag_group", "time_in_hospital", "num_medications",
                   "number_inpatient", "number_diagnoses", "discharge_group"]].copy()
    d.insert(0, "risk_30d", p.round(4))
    d.insert(1, "tier", [tier(x) for x in p])
    d["actual_readmitted"] = s.test_df["readmitted_30d"].values
    d.index.name = "patient_id"
    return d.sort_values("risk_30d", ascending=False)


def load_json(path):
    return json.load(open(path)) if path.exists() else None


if not (ARTIFACTS / "bundle.joblib").exists():
    st.error("No trained models found. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()

agent = load_agent()
sysm = agent.sys
cohort = score_cohort()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 CareCompass")
page = st.sidebar.radio("Navigate", ["Overview", "Triage list", "Patient workspace", "Review queue"])
st.sidebar.divider()
st.sidebar.caption(f"Agent mode: **{agent.mode}**")
st.sidebar.caption("Decision support only. Not a medical device; every output requires clinician review.")

# ================================================================== 1. OVERVIEW
if page == "Overview":
    st.title("Hospital readmission risk: model overview")
    m = load_json(REPORTS / "metrics.json")
    if not m:
        st.warning("reports/metrics.json not found.")
        st.stop()
    st.caption(f"Data source: {m['data_source']}")
    ens = m["test"]["Ensemble ML+DL"]
    c = st.columns(5)
    c[0].metric("Test ROC-AUC", f"{ens['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{ens['pr_auc']:.3f}", help="Random guessing scores about the base rate (~0.10).")
    c[2].metric("Recall", f"{ens['recall']:.0%}", help="Share of true readmissions the model flags.")
    c[3].metric("Precision", f"{ens['precision']:.0%}")
    c[4].metric("Decision threshold", f"{m['threshold']:.2f}")

    st.subheader("Model comparison (held-out test set)")
    st.dataframe(pd.DataFrame(m["test"]).T.round(3), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Cross-validated model selection")
        cv = pd.DataFrame(m["ml_cv"]).T[["cv_pr_auc_mean", "cv_pr_auc_std", "val_roc_auc"]].round(3)
        st.dataframe(cv, width="stretch")
        st.caption("Simple models often win on noisy tabular data; the pipeline picks by CV, not by prestige.")
    with right:
        st.subheader("Generative model: synthetic patients (cVAE)")
        g = m["cvae_augmentation"]
        st.metric("Feature-mean correlation with real high-risk patients", f"{g['fidelity_corr']:.3f}")
        st.write(f"Test PR-AUC without synthetic data **{g['baseline']['pr_auc']:.3f}**, "
                 f"with **{g['augmented']['pr_auc']:.3f}**. Realistic, but no accuracy gain here.")

    if (REPORTS / "model_report.png").exists():
        st.subheader("ROC, precision-recall, calibration, feature importance")
        st.image(str(REPORTS / "model_report.png"), width="stretch")

    st.subheader("Fairness audit")
    fa = pd.read_csv(REPORTS / "fairness_audit.csv")
    attr = st.selectbox("Attribute", fa["attribute"].unique())
    sub = fa[fa["attribute"] == attr].set_index("group")
    st.bar_chart(sub[["recall@thr", "auc"]])
    st.dataframe(sub, width="stretch")
    worst = sub["recall@thr"].idxmin()
    if sub["recall@thr"].max() - sub["recall@thr"].min() > 0.15:
        st.warning(f"Recall gap across {attr}: lowest for **{worst}** ({sub.loc[worst, 'recall@thr']:.0%}). "
                   "Needs review before any deployment.")

    st.subheader("Patient segments (PCA + K-Means)")
    st.dataframe(pd.read_csv(REPORTS / "patient_segments.csv").set_index("segment"), width="stretch")

# ================================================================== 2. TRIAGE LIST
elif page == "Triage list":
    st.title("Discharge triage list")
    f = st.columns(4)
    tiers = f[0].multiselect("Risk tier", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    diag = f[1].multiselect("Diagnosis group", sorted(cohort["diag_group"].unique()))
    age = f[2].slider("Age", 15, 100, (15, 100))
    gender = f[3].multiselect("Gender", sorted(cohort["gender"].unique()))
    v = cohort[cohort["tier"].isin(tiers) & cohort["age_num"].between(*age)]
    if diag: v = v[v["diag_group"].isin(diag)]
    if gender: v = v[v["gender"].isin(gender)]

    c = st.columns(4)
    c[0].metric("Patients shown", f"{len(v):,}")
    c[1].metric("Avg predicted risk", f"{v['risk_30d'].mean():.1%}" if len(v) else "n/a")
    c[2].metric("Actually readmitted", f"{v['actual_readmitted'].mean():.1%}" if len(v) else "n/a",
                help="Observed outcome in the held-out test set, shown to validate the model.")
    c[3].metric("Cohort base rate", f"{cohort['actual_readmitted'].mean():.1%}")

    bucket = pd.cut(cohort["risk_30d"], [0, .05, .10, .15, .20, .30, 1], include_lowest=True)
    st.subheader("Does a higher score mean more real readmissions?")
    st.bar_chart(cohort.groupby(bucket, observed=True)["actual_readmitted"].mean().rename("observed readmission rate")
                 .rename_axis("predicted risk band").reset_index().assign(
                     **{"predicted risk band": lambda d: d["predicted risk band"].astype(str)}).set_index("predicted risk band"))
    st.subheader("Patients")
    st.dataframe(v.head(500), width="stretch", height=420)
    st.caption("Open a patient in the Patient workspace page using their patient_id.")

# ================================================================== 3. PATIENT WORKSPACE
elif page == "Patient workspace":
    st.title("Patient workspace")
    default = int(cohort.index[0])
    pid = int(st.number_input("patient_id", 0, len(sysm.test_df) - 1, st.session_state.get("pid", default), step=1))
    st.session_state["pid"] = pid
    rec = sysm.get_patient_record(pid)
    pred = sysm.predict_readmission_risk(pid)

    h = st.columns(4)
    h[0].metric("30-day readmission risk", f"{pred['risk_30d']:.1%}", f"{pred['risk_30d'] - pred['cohort_base_rate']:+.1%} vs cohort",
                delta_color="inverse")
    h[1].metric("Risk tier", f"{TIER_COLOR[pred['tier']]} {pred['tier']}")
    h[2].metric("Model uncertainty", f"±{pred['uncertainty']:.1%}", help="MC-dropout spread + ML/deep-net disagreement.")
    h[3].metric("ML vs deep net", f"{pred['ml_model_prob']:.1%} / {pred['deep_net_prob']:.1%}")
    st.progress(min(pred["risk_30d"] / 0.5, 1.0), text=f"Risk scale: low < {TIER_MED:.0%} < medium < {TIER_HIGH:.0%} < high")

    a, b = st.columns(2)
    with a:
        st.subheader("Why this score? (top drivers)")
        drv = pd.DataFrame(sysm.explain_risk_drivers(pid)["drivers"])
        drv["label"] = drv["feature"] + " = " + drv["value"].astype(str)
        st.bar_chart(drv.set_index("label")["effect_on_risk"])
        st.caption("Positive bars push risk up, negative bars push it down (vs. a typical patient).")
        st.subheader("Record")
        st.dataframe(pd.Series(rec, name="value").astype(str), width="stretch", height=300)
    with b:
        st.subheader("Similar past patients")
        sim = sysm.find_similar_patients(pid)
        st.metric("Readmission rate among 30 nearest neighbours", f"{sim['neighbor_readmission_rate']:.0%}")
        st.write(f"Segment **{sim['segment_id']}** profile:")
        st.dataframe(pd.Series(sim["segment_profile"], name="segment avg").round(2), width="stretch")
        st.subheader("Data-quality check")
        an = sysm.check_data_anomaly(pid)
        (st.warning if "atypical" in an["interpretation"] else st.success)(
            f"{an['interpretation'].capitalize()} (autoencoder percentile {an['autoencoder_error_percentile']:.0%}, "
            f"isolation-forest outlier: {an['isolation_forest_outlier']})")
        actual = int(sysm.test_df.iloc[pid]["readmitted_30d"])
        st.caption(f"Ground truth (test set only): {'readmitted within 30 days' if actual else 'not readmitted within 30 days'}")

    st.divider()
    st.subheader("🤖 Discharge-planning agent")
    st.write("The agent calls the tools above, searches clinical guidelines, drafts a plan, passes it through a safety "
             "reviewer, and escalates to a human when needed.")
    if st.button("Run agent for this patient", type="primary"):
        with st.spinner("Agent working..."):
            agent.memory.pop(pid, None)
            st.session_state[f"res{pid}"] = agent.run(pid, verbose=False)
    res = st.session_state.get(f"res{pid}")
    if res:
        st.caption(f"Mode: {res['mode']}")
        steps = " → ".join(t["tool"] for t in res["trace"])
        st.info(f"**Trace:** {steps}")
        (st.error if res["flagged"] else st.success)(
            "🚩 Escalated to human clinician review (see Review queue)" if res["flagged"] else "No escalation needed")
        t1, t2, t3 = st.tabs(["Clinician note", "Patient message", "Tool call details"])
        t1.markdown(res["clinician_note"].replace("\n", "  \n"))
        t2.write(res["patient_message"])
        with t3:
            for t in res["trace"]:
                with st.expander(t["tool"]):
                    st.json({"args": t["args"], "result": t["result"]})

# ================================================================== 4. REVIEW QUEUE
else:
    st.title("Human review queue")
    q = REPORTS / "review_queue.jsonl"
    if not q.exists() or q.stat().st_size == 0:
        st.info("No cases escalated yet. Run the agent on a high-risk patient from the Patient workspace.")
    else:
        rows = pd.DataFrame([json.loads(l) for l in q.read_text().splitlines() if l.strip()])
        rows = rows.sort_values(["urgency", "ts"], ascending=[True, False])
        c = st.columns(3)
        c[0].metric("Cases queued", len(rows))
        c[1].metric("Urgent", int((rows["urgency"] == "urgent").sum()))
        c[2].metric("Distinct patients", rows["patient_id"].nunique())
        st.dataframe(rows, width="stretch")
        if st.button("Clear queue"):
            q.write_text("")
            st.rerun()
    log = ARTIFACTS / "agent_log.jsonl"
    if log.exists():
        st.subheader("Agent audit log")
        st.dataframe(pd.DataFrame([json.loads(l) for l in log.read_text().splitlines() if l.strip()]).iloc[::-1],
                     width="stretch")
