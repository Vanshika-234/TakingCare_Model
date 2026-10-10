"""Classical machine learning layer.

Covers: supervised learning (linear, bagging, boosting), cross-validation, class imbalance,
probability calibration, threshold selection (cost-sensitive), model-agnostic explainability
(permutation importance + per-patient occlusion), unsupervised learning (PCA, K-Means,
Isolation Forest), and a subgroup fairness audit.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import SEED
from src.data import CAT_COLS, NUM_COLS, TARGET


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                            max_iter=250, l2_regularization=1.0,
                                                            class_weight="balanced", random_state=SEED),
    }


def metrics(y, p, thr=0.5) -> dict:
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "recall": recall_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred), "brier": brier_score_loss(y, p)}


def best_threshold(y, p, beta=2.0) -> float:
    """Pick the threshold maximizing F-beta (beta=2 weights recall: a missed readmission costs more
    than an unnecessary follow-up call)."""
    grid = np.linspace(0.02, 0.9, 89)
    scores = []
    for t in grid:
        pr, rc = precision_score(y, p >= t, zero_division=0), recall_score(y, p >= t)
        scores.append(0 if pr + rc == 0 else (1 + beta**2) * pr * rc / (beta**2 * pr + rc))
    return float(grid[int(np.argmax(scores))])


def train_supervised(Xtr, ytr, Xva, yva):
    """Cross-validate every candidate, pick the best by CV PR-AUC, calibrate it on the validation set."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report, fitted = {}, {}
    for name, model in candidate_models().items():
        cvs = cross_val_score(model, Xtr, ytr, cv=cv, scoring="average_precision", n_jobs=1)
        model.fit(Xtr, ytr)
        fitted[name] = model
        report[name] = {"cv_pr_auc_mean": float(cvs.mean()), "cv_pr_auc_std": float(cvs.std()),
                        **{f"val_{k}": v for k, v in metrics(yva, model.predict_proba(Xva)[:, 1]).items()}}
    best = max(report, key=lambda k: report[k]["cv_pr_auc_mean"])
    # Balanced class weights distort probabilities -> calibrate so "0.18" really means ~18%.
    cal = CalibratedClassifierCV(FrozenEstimator(fitted[best]), method="isotonic")
    cal.fit(Xva, yva)
    return fitted, cal, best, report


def global_importance(model, X, y, names, n_repeats=5, max_rows=4000):
    idx = np.random.default_rng(SEED).choice(len(X), min(max_rows, len(X)), replace=False)
    r = permutation_importance(model, X[idx], y[idx], scoring="average_precision",
                               n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"feature": names, "importance": r.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def explain_patient(predict_fn, preprocessor, row: pd.DataFrame, reference: pd.DataFrame, top=6):
    """Model-agnostic local explanation by occlusion: replace one raw feature at a time with the
    cohort median/mode and measure the drop in predicted risk (positive = pushes risk UP)."""
    base = float(predict_fn(preprocessor.transform(row))[0])
    out = []
    for col in NUM_COLS + CAT_COLS:
        alt = row.copy()
        alt[col] = reference[col].median() if col in NUM_COLS else reference[col].mode().iloc[0]
        delta = base - float(predict_fn(preprocessor.transform(alt))[0])
        out.append((col, row[col].iloc[0], delta))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return base, [{"feature": f, "value": v if not isinstance(v, (np.floating, float)) else round(float(v), 2),
                   "effect_on_risk": round(d, 4)} for f, v, d in out[:top]]


# ---------------------------------------------------------------- unsupervised learning
def fit_patient_segments(Xtr, df_tr, k_range=range(3, 8)):
    """PCA -> K-Means with silhouette-based k. Returns models + a clinical profile of each segment."""
    pca = PCA(n_components=0.90, random_state=SEED).fit(Xtr)
    Z = pca.transform(Xtr)
    sub = np.random.default_rng(SEED).choice(len(Z), min(5000, len(Z)), replace=False)
    sil = {k: silhouette_score(Z[sub], KMeans(k, n_init=5, random_state=SEED).fit_predict(Z[sub])) for k in k_range}
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    prof = df_tr.assign(segment=km.labels_).groupby("segment").agg(
        n=("segment", "size"), readmit_rate=(TARGET, "mean"), age=("age_num", "mean"),
        los=("time_in_hospital", "mean"), meds=("num_medications", "mean"),
        prior_inpatient=("number_inpatient", "mean"), diagnoses=("number_diagnoses", "mean")).round(3)
    return pca, km, prof, sil


def fit_anomaly_detector(Xtr):
    return IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED).fit(Xtr)


# ---------------------------------------------------------------- fairness
def fairness_audit(df_te, p, thr):
    rows = []
    for col in ["gender", "race", "age_bucket"]:
        d = df_te.assign(p=p, age_bucket=pd.cut(df_te["age_num"], [0, 45, 65, 80, 120],
                                                labels=["<45", "45-64", "65-79", "80+"]))
        for g, sub in d.groupby(col, observed=True):
            if len(sub) < 100 or sub[TARGET].nunique() < 2:
                continue
            rows.append({"attribute": col, "group": str(g), "n": len(sub),
                         "base_rate": round(sub[TARGET].mean(), 3),
                         "auc": round(roc_auc_score(sub[TARGET], sub["p"]), 3),
                         "recall@thr": round(recall_score(sub[TARGET], sub["p"] >= thr), 3),
                         "flag_rate": round((sub["p"] >= thr).mean(), 3)})
    return pd.DataFrame(rows)
