"""CareSystem: loads every trained artifact and exposes them as clean, JSON-returning *tools*
that the agent can call. Each tool = one capability from the ML / DL / GenAI layers."""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from config import ARTIFACTS, REPORTS, TIER_HIGH, TIER_MED
from src import dl_models as dl
from src.data import CAT_COLS, NUM_COLS, TARGET
from src.gen_models import GuidelineRAG
from src.ml_models import explain_patient

FEATURES = NUM_COLS + CAT_COLS


def tier(p):
    return "HIGH" if p >= TIER_HIGH else "MEDIUM" if p >= TIER_MED else "LOW"


class CareSystem:
    def __init__(self):
        b = joblib.load(ARTIFACTS / "bundle.joblib")
        self.__dict__.update(b)                                   # pre, cal, dl_iso, km, pca, iso_forest, ...
        self.net = dl.RiskNet(b["d_in"]); self.net.load_state_dict(b["net_state"]); self.net.eval()
        self.ae = dl.AutoEncoder(b["d_in"]); self.ae.load_state_dict(b["ae_state"]); self.ae.eval()
        self.rag = GuidelineRAG()
        self.nn = NearestNeighbors(n_neighbors=30).fit(b["train_embed"])
        self.review_queue = REPORTS / "review_queue.jsonl"

    # ---------- shared helpers
    def _row(self, pid):
        return self.test_df.iloc[[int(pid)]][FEATURES]

    def _predict(self, row):
        X = self.pre.transform(row)
        p_ml = float(self.cal.predict_proba(X)[0, 1])
        m, s = dl.mc_dropout_predict(self.net, X)
        p_dl = float(self.dl_iso.predict(m)[0])
        risk = (p_ml + p_dl) / 2
        unc = float(np.sqrt(s[0] ** 2 + ((p_ml - p_dl) / 2) ** 2))   # MC-dropout spread + model disagreement
        return risk, unc, p_ml, p_dl

    def _ensemble_fn(self, X):
        p_ml = self.cal.predict_proba(X)[:, 1]
        p_dl = self.dl_iso.predict(dl.predict_proba(self.net, X))
        return (p_ml + p_dl) / 2

    # ---------- TOOLS (each returns a JSON-serialisable dict)
    def get_patient_record(self, patient_id: int):
        r = self.test_df.iloc[int(patient_id)]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in r[FEATURES].items()}

    def predict_readmission_risk(self, patient_id: int):
        risk, unc, p_ml, p_dl = self._predict(self._row(patient_id))
        return {"risk_30d": round(risk, 4), "uncertainty": round(unc, 4), "tier": tier(risk),
                "ml_model_prob": round(p_ml, 4), "deep_net_prob": round(p_dl, 4),
                "cohort_base_rate": round(self.base_rate, 4)}

    def explain_risk_drivers(self, patient_id: int):
        base, drivers = explain_patient(self._ensemble_fn, self.pre, self._row(patient_id), self.ref_df)
        return {"risk_30d": round(base, 4), "drivers": drivers,
                "note": "effect_on_risk = change in probability vs. replacing the feature with the cohort typical value"}

    def find_similar_patients(self, patient_id: int, k: int = 30):
        X = self.pre.transform(self._row(patient_id))
        _, idx = self.nn.kneighbors(dl.embed(self.ae, X), n_neighbors=min(k, 30))
        y = self.train_y[idx[0]]
        seg = int(self.km.predict(self.pca.transform(X))[0])
        return {"n_neighbors": len(y), "neighbor_readmission_rate": round(float(y.mean()), 3),
                "segment_id": seg, "segment_profile": self.seg_profile.loc[seg].to_dict()}

    def check_data_anomaly(self, patient_id: int):
        X = self.pre.transform(self._row(patient_id))
        err = float(dl.reconstruction_error(self.ae, X)[0])
        pct = float((self.train_recon_err < err).mean())
        iso = int(self.iso_forest.predict(X)[0] == -1)
        return {"autoencoder_error_percentile": round(pct, 3), "isolation_forest_outlier": bool(iso),
                "interpretation": "atypical record - verify data quality" if (pct > 0.98 or iso) else "typical record"}

    def search_guidelines(self, query: str, k: int = 3):
        return {"results": self.rag.search(query, k)}

    def flag_for_clinician_review(self, patient_id: int, reason: str, urgency: str = "routine"):
        """ACTION tool (has a side-effect): queue the case for a human. The agent may NOT change care itself."""
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "patient_id": int(patient_id),
               "reason": reason, "urgency": urgency}
        with open(self.review_queue, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return {"status": "queued_for_human_review", **rec}

    def call(self, name: str, args: dict):
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            # round-trip through JSON so every tool result is plain python (numpy scalars -> int/float)
            return json.loads(json.dumps(fn(**args), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        except Exception as e:  # tool errors go back to the agent, not up the stack
            return {"error": f"{type(e).__name__}: {e}"}


TOOL_SCHEMAS = [
    {"name": "get_patient_record", "description": "Fetch the discharge record (demographics, utilisation, diagnosis group, meds) for a patient.",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}}, "required": ["patient_id"]}},
    {"name": "predict_readmission_risk", "description": "Calibrated 30-day readmission probability from an ML + deep-learning ensemble, with uncertainty and risk tier.",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}}, "required": ["patient_id"]}},
    {"name": "explain_risk_drivers", "description": "Top features pushing this patient's risk up or down (occlusion-based local explanation).",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}}, "required": ["patient_id"]}},
    {"name": "find_similar_patients", "description": "Find similar past patients (autoencoder embedding kNN + K-Means segment) and their observed readmission rate.",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}, "k": {"type": "integer"}}, "required": ["patient_id"]}},
    {"name": "check_data_anomaly", "description": "Autoencoder + Isolation Forest check that the record is typical and the prediction can be trusted.",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}}, "required": ["patient_id"]}},
    {"name": "search_guidelines", "description": "Retrieve relevant clinical guideline passages (RAG). Cite results as [G#].",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "flag_for_clinician_review", "description": "Queue the patient for human clinician review. Use for HIGH risk, high uncertainty, or anomalous records.",
     "input_schema": {"type": "object", "properties": {"patient_id": {"type": "integer"}, "reason": {"type": "string"},
                                                       "urgency": {"type": "string", "enum": ["routine", "urgent"]}},
                      "required": ["patient_id", "reason"]}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
