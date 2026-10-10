# CareCompass: Hospital Readmission Risk & Discharge-Planning Agent

**Real-world problem.** About 1 in 5 Medicare patients is readmitted within 30 days of discharge. These
readmissions are costly, harmful, and often preventable, and U.S. hospitals are financially penalised for them.
Care teams can't give every patient intensive follow-up, so they need to know **who** is at risk, **why**, and
**what to do about it**, in a form they can trust.

**Dataset.** [UCI "Diabetes 130-US Hospitals, 1999-2008"](https://archive.ics.uci.edu/dataset/296) (~100k encounters,
50 features). Download `diabetic_data.csv` into `data/` to use it. With no file present, the project generates a
synthetic cohort with the *identical schema* so everything runs out of the box.

## Architecture

```
          ┌───────────────────────── AGENTIC AI ─────────────────────────┐
          │  Discharge Planning Agent (ReAct loop, memory, guardrails)   │
          │  plan → tool calls → observe → draft → SAFETY REVIEW → output│
          └──┬──────────┬───────────┬───────────┬───────────┬───────────┬┘
   predict_risk   explain_drivers  find_similar  check_anomaly  search_guidelines  flag_for_review
          │          │           │           │           │            │(action)
   ┌──────┴──────────┴───┐  ┌────┴───────────┴────┐  ┌───┴──────────────────┐
   │ MACHINE LEARNING    │  │ DEEP LEARNING       │  │ GENERATIVE AI        │
   │ LogReg/RF/GBM + CV  │  │ RiskNet MLP         │  │ Conditional VAE      │
   │ calibration, F2 thr │  │ MC-dropout          │  │ RAG over guidelines  │
   │ PCA + K-Means       │  │ Autoencoder         │  │ LLM notes (grounded) │
   │ IsolationForest     │  │ (embeddings/anomaly)│  │                      │
   └─────────────────────┘  └─────────────────────┘  └──────────────────────┘
```

## Concept coverage

| Area | Concepts implemented | Where |
|---|---|---|
| **Machine learning** | Preprocessing pipelines (imputation, scaling, one-hot), feature engineering, stratified splits, class imbalance (class weights), logistic regression / random forest / gradient boosting, k-fold cross-validation, model selection, **probability calibration** (isotonic), cost-sensitive thresholding (F2), permutation importance, local occlusion explanations, PCA, K-Means + silhouette, Isolation Forest, **subgroup fairness audit** | `src/ml_models.py`, `src/data.py` |
| **Deep learning** | MLP with BatchNorm, Dropout, weight decay (AdamW), weighted BCE, LR scheduling, gradient clipping, early stopping, **Monte-Carlo-dropout uncertainty**, **autoencoder** (embeddings + reconstruction-error anomaly detection) | `src/dl_models.py` |
| **Generative AI** | **Conditional VAE** (reparameterisation trick, KL term) for synthetic patients; **RAG** (TF-IDF retrieval over guidelines, relevance floor); grounded LLM generation with citations; audience-specific prompting (clinician vs patient at a 6th-grade level); offline template fallback | `src/gen_models.py` |
| **Agentic AI** | Tool/function calling (7 tools), ReAct loop with step budget, explicit plan, session memory + persistent audit log, **reflection via a Safety Reviewer**, **human-in-the-loop escalation** (action tool writes to a review queue), guardrails (no medication changes, mandatory disclaimer), graceful degradation (live Claude ↔ rule-based planner) | `src/agent.py`, `src/system.py` |

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py            # trains everything (~1 min on CPU) -> artifacts/, reports/
python demo_agent.py --top 3      # agent on the 3 highest-risk test patients
python demo_agent.py 0 1 2        # or specific patients

export ANTHROPIC_API_KEY=...      # optional: Claude drives the tools and writes the notes
```

### Dashboard

```bash
streamlit run dashboard.py        # opens http://localhost:8501
```

| Page | What it shows |
|---|---|
| Overview | Test metrics, model comparison, calibration/ROC plots, fairness audit by group, patient segments, generative-model check |
| Triage list | Every patient ranked by risk, with filters (tier, diagnosis, age, gender) and a chart checking that higher scores mean more real readmissions |
| Patient workspace | One patient: risk and uncertainty, top drivers, similar past patients, data-quality check, and a button to run the agent (trace, clinician note, patient message, tool details) |
| Review queue | Cases the agent escalated to a human, plus the agent audit log |

Outputs: `reports/model_report.png` (ROC / PR / calibration / importance), `reports/metrics.json`,
`reports/fairness_audit.csv`, `reports/patient_segments.csv`, `reports/review_queue.jsonl`.

## Results on the bundled synthetic cohort (held-out test set)

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Logistic regression (calibrated) | 0.645 | 0.170 | 0.73 | 0.14 |
| Deep net (calibrated) | 0.639 | 0.164 | 0.72 | 0.13 |
| **Ensemble** | **0.646** | **0.173** | 0.69 | 0.14 |

Base rate is ~10%. An AUC around 0.65 matches what published models achieve on real readmission data,
because readmission is inherently noisy. Treat these numbers as a demonstration of the pipeline; re-run on the
real UCI file for real-data performance.

## Honest findings and limitations

* **Simple beat complex.** Logistic regression won cross-validation over the random forest and gradient
  boosting, and the deep net did not beat it. On modest tabular data with a noisy outcome this is common and expected.
* **cVAE augmentation did not help** (test PR-AUC 0.178 → 0.178), though the synthetic patients matched the real
  high-risk feature means (r = 0.997). Generative augmentation is shown as a technique, not claimed as a win.
* **The fairness audit surfaced a gap.** Recall is only ~0.13 for patients under 45 versus 0.66 to 0.86 for older
  groups, because the model leans on age and prior stays. Before any deployment this would need subgroup
  thresholds or re-weighting.
* **Synthetic data has no real clinical signal.** The synthetic cohort's risk function was written by hand; it
  exercises the code but proves nothing clinically.
* **The live (Claude) agent path** was tested for mechanics (tool loop, critic, revision) with a scripted fake
  client, not against the real API.
* The guideline knowledge base is a short illustrative set, not medical advice. This is a research and
  teaching project, not a medical device. All outputs require clinician review.

## Ideas to extend

Replace TF-IDF with dense embeddings; add SHAP; train a text model on discharge notes; add a Streamlit UI;
evaluate agent behaviour with a scripted test-suite of patients; add temporal validation (train on early years,
test on later ones).
