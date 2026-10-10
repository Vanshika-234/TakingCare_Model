"""Train every model in CareCompass and save a single bundle for the agent.

    python run_pipeline.py
"""
import json
import time

import joblib
import matplotlib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ARTIFACTS, REPORTS, SEED
from src import dl_models as dl
from src import ml_models as ml
from src.data import CAT_COLS, NUM_COLS, TARGET, build_preprocessor, get_splits
from src.gen_models import train_cvae

t0 = time.time()
step = lambda s: print(f"\n[{time.time() - t0:5.0f}s] {s}")

# ------------------------------------------------------------------ 1. data
step("1/7 Data")
S = get_splits()
df, tr, va, te = S["df"], S["train"], S["val"], S["test"]
print(f"   source: {S['source']}\n   rows={len(df):,}  readmit<30d rate={df[TARGET].mean():.1%}")
pre = build_preprocessor().fit(df.loc[tr, NUM_COLS + CAT_COLS])
Xtr, Xva, Xte = (pre.transform(df.loc[i, NUM_COLS + CAT_COLS]) for i in (tr, va, te))
ytr, yva, yte = (df.loc[i, TARGET].values for i in (tr, va, te))
names = list(pre.get_feature_names_out())
print(f"   features after preprocessing: {Xtr.shape[1]}")

# ------------------------------------------------------------------ 2. classical ML
step("2/7 Classical ML (cross-validated model selection + calibration)")
fitted, cal, best, ml_report = ml.train_supervised(Xtr, ytr, Xva, yva)
for k, v in ml_report.items():
    print(f"   {k:20s} CV PR-AUC {v['cv_pr_auc_mean']:.3f}±{v['cv_pr_auc_std']:.3f} | val ROC-AUC {v['val_roc_auc']:.3f}")
print(f"   -> selected: {best} (isotonic-calibrated)")
p_ml_va, p_ml_te = cal.predict_proba(Xva)[:, 1], cal.predict_proba(Xte)[:, 1]
imp = ml.global_importance(fitted[best], Xte, yte, names)
print("   top global drivers:", ", ".join(imp.feature.head(5)))

step("   Unsupervised: PCA + K-Means patient segments, Isolation Forest")
pca, km, seg_profile, sil = ml.fit_patient_segments(Xtr, df.loc[tr])
print(f"   silhouette by k: { {k: round(v, 3) for k, v in sil.items()} } -> k={km.n_clusters}")
print(seg_profile.to_string())
iso_forest = ml.fit_anomaly_detector(Xtr)

# ------------------------------------------------------------------ 3. deep learning
step("3/7 Deep learning (RiskNet MLP + Autoencoder)")
net, hist = dl.train_risknet(Xtr, ytr, Xva, yva)
dl_iso = IsotonicRegression(out_of_bounds="clip").fit(dl.predict_proba(net, Xva), yva)
p_dl_va, p_dl_te = dl_iso.predict(dl.predict_proba(net, Xva)), dl_iso.predict(dl.predict_proba(net, Xte))
ae = dl.train_autoencoder(Xtr)
train_embed, train_err = dl.embed(ae, Xtr), dl.reconstruction_error(ae, Xtr)

# ------------------------------------------------------------------ 4. generative: cVAE augmentation
step("4/7 Generative AI: conditional VAE synthetic high-risk patients")
vae = train_cvae(Xtr, ytr)
n_syn = int((ytr == 0).sum() * 0.5 - (ytr == 1).sum())            # raise positives to ~1/3 of data
Xsyn = vae.sample(max(n_syn, 0), 1.0)
aug = clone(fitted[best]).fit(np.vstack([Xtr, Xsyn]), np.r_[ytr, np.ones(len(Xsyn))])
base_m = ml.metrics(yte, fitted[best].predict_proba(Xte)[:, 1], 0.5)
aug_m = ml.metrics(yte, aug.predict_proba(Xte)[:, 1], 0.5)
real_pos_mean, syn_mean = Xtr[ytr == 1].mean(0), Xsyn.mean(0)
fidelity = float(np.corrcoef(real_pos_mean, syn_mean)[0, 1])
print(f"   generated {len(Xsyn):,} synthetic patients | feature-mean correlation with real positives: {fidelity:.3f}")
print(f"   test PR-AUC  baseline {base_m['pr_auc']:.3f}  vs  +synthetic {aug_m['pr_auc']:.3f}")

# ------------------------------------------------------------------ 5. evaluation
step("5/7 Held-out test evaluation")
p_ens_va, p_ens_te = (p_ml_va + p_dl_va) / 2, (p_ml_te + p_dl_te) / 2
thr = ml.best_threshold(yva, p_ens_va)
results = {"ML (" + best + ", calibrated)": p_ml_te, "Deep net (calibrated)": p_dl_te, "Ensemble ML+DL": p_ens_te}
final = {}
for name, p in results.items():
    final[name] = ml.metrics(yte, p, thr)
    m = final[name]
    print(f"   {name:34s} ROC-AUC {m['roc_auc']:.3f} PR-AUC {m['pr_auc']:.3f} recall {m['recall']:.2f} "
          f"precision {m['precision']:.2f} Brier {m['brier']:.3f}")
print(f"   decision threshold (F2-optimal on validation): {thr:.2f}")
step("   Fairness audit (ensemble)")
fair = ml.fairness_audit(df.loc[te], p_ens_te, thr)
print(fair.to_string(index=False))

# ------------------------------------------------------------------ 6. plots
step("6/7 Reports")
fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
for name, p in results.items():
    f, t, _ = roc_curve(yte, p); ax[0].plot(f, t, label=f"{name.split(' (')[0]} ({final[name]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=name.split(" (")[0])
    fp, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile"); ax[2].plot(mp, fp, "o-", label=name.split(" (")[0])
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=8)
ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
ax[2].plot([0, .5], [0, .5], "k:"); ax[2].set(title="Calibration", xlabel="Predicted", ylabel="Observed")
top = imp.head(10)[::-1]; ax[3].barh(top.feature.str.replace(r"^(num|cat)__", "", regex=True), top.importance)
ax[3].set(title="Permutation importance (PR-AUC drop)")
plt.tight_layout(); plt.savefig(REPORTS / "model_report.png", dpi=110); plt.close()
plt.figure(figsize=(5, 3.5)); plt.plot(hist); plt.title("RiskNet validation loss"); plt.xlabel("epoch")
plt.tight_layout(); plt.savefig(REPORTS / "training_curve.png", dpi=110); plt.close()
json.dump({"data_source": S["source"], "ml_cv": ml_report, "test": final, "threshold": thr,
           "cvae_augmentation": {"baseline": base_m, "augmented": aug_m, "fidelity_corr": fidelity}},
          open(REPORTS / "metrics.json", "w"), indent=2, default=float)
fair.to_csv(REPORTS / "fairness_audit.csv", index=False)
seg_profile.to_csv(REPORTS / "patient_segments.csv")

# ------------------------------------------------------------------ 7. bundle for the agent
step("7/7 Saving bundle")
joblib.dump({"pre": pre, "cal": cal, "dl_iso": dl_iso, "km": km, "pca": pca, "iso_forest": iso_forest,
             "d_in": Xtr.shape[1], "net_state": net.state_dict(), "ae_state": ae.state_dict(),
             "train_embed": train_embed, "train_y": ytr, "train_recon_err": train_err,
             "seg_profile": seg_profile, "test_df": df.loc[te].reset_index(drop=True),
             "ref_df": df.loc[tr, NUM_COLS + CAT_COLS], "base_rate": float(ytr.mean()), "threshold": thr},
            ARTIFACTS / "bundle.joblib")
print("   done. Try:  python demo_agent.py 0 1 2")
