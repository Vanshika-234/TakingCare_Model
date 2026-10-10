"""Generative AI layer.

1. Conditional VAE  -> generative model of patient feature vectors; synthesizes realistic
   high-risk patients to rebalance training data (and enables privacy-friendlier data sharing).
2. RAG              -> retrieval over a clinical-guideline knowledge base (TF-IDF + cosine),
   so LLM output is grounded in cited sources rather than free recall.
3. LLM client       -> Claude for patient-friendly explanations / clinician summaries, with a
   deterministic template fallback so the project works with no API key.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

from config import LLM_MODEL, SEED


# ----------------------------------------------------------------------------- 1. cVAE
class CVAE(nn.Module):
    def __init__(self, d_in, latent=12, n_cond=1):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(d_in + n_cond, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mu, self.logvar = nn.Linear(32, latent), nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent + n_cond, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, d_in))

    def forward(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, lv = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)       # reparameterization trick
        return self.dec(torch.cat([z, c], 1)), mu, lv

    @torch.no_grad()
    def sample(self, n, cond_value=1.0):
        z = torch.randn(n, self.latent)
        c = torch.full((n, 1), cond_value)
        return self.dec(torch.cat([z, c], 1)).numpy()


def train_cvae(X, y, epochs=40, beta=0.5):
    torch.manual_seed(SEED)
    m = CVAE(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                  torch.tensor(y, dtype=torch.float32).unsqueeze(1)), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for xb, cb in dl:
            opt.zero_grad()
            rec, mu, lv = m(xb, cb)
            recon = nn.functional.mse_loss(rec, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            (recon + beta * kl).backward()
            opt.step()
    m.eval()
    return m


# ----------------------------------------------------------------------------- 2. RAG
GUIDELINES = [
    ("Transitional care: follow-up call", "Contact high-risk discharged patients within 48 to 72 hours of discharge to review symptoms, medications, and appointments. Early telephone follow-up reduces avoidable 30-day readmissions."),
    ("Transitional care: early clinic visit", "Schedule a primary-care or specialist visit within 7 days of discharge for high-risk patients, and within 14 days for moderate-risk patients."),
    ("Medication reconciliation", "Perform medication reconciliation at discharge. Patients on 15 or more medications (polypharmacy) benefit from pharmacist review for duplications, interactions, and deprescribing opportunities."),
    ("Diabetes: glycemic management", "For patients with diabetes and HbA1c above 8 percent, review insulin regimen before discharge, provide hypoglycemia education, and arrange diabetes educator follow-up. Avoid abrupt regimen changes without a monitoring plan."),
    ("Heart failure discharge bundle", "For circulatory or heart-failure admissions: daily weights, sodium restriction teaching, diuretic plan, and a cardiology or heart-failure clinic visit within 7 days."),
    ("Respiratory admissions", "For COPD or pneumonia admissions: confirm inhaler technique, vaccination status, and provide an action plan for worsening symptoms; consider pulmonary rehabilitation referral."),
    ("Repeated hospitalizations", "Patients with two or more inpatient stays in the prior year are at the highest readmission risk; consider case management enrollment, home health, or hospital-at-home programs."),
    ("Discharge to skilled nursing facility", "When discharging to a facility, send a complete handoff including medication list, pending tests, and goals of care, and name a clinical contact for questions."),
    ("Teach-back and health literacy", "Use teach-back to confirm patient understanding. Provide written instructions at a sixth-grade reading level and include warning signs that should prompt a call or emergency visit."),
    ("Social determinants of health", "Screen for transportation, food, housing, and caregiver barriers. Unresolved social needs are a common hidden driver of missed follow-up and readmission."),
    ("Long length of stay", "Extended stays (more than 7 days) indicate complexity and deconditioning; assess functional status, fall risk, and need for physical therapy or home services at discharge."),
    ("AI decision-support safety", "Model risk scores are decision support, not diagnoses. A clinician must review any high-risk flag, and the model must not autonomously change medications or treatment. Audit performance across age, sex, and race for equity."),
]


class GuidelineRAG:
    def __init__(self):
        self.titles = [t for t, _ in GUIDELINES]
        self.texts = [f"{t}. {b}" for t, b in GUIDELINES]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(self.texts)
        self.M = self.vec.transform(self.texts)

    def search(self, query: str, k=3):
        s = cosine_similarity(self.vec.transform([query]), self.M).ravel()
        return [{"source": f"G{i + 1}: {self.titles[i]}", "text": GUIDELINES[i][1], "score": round(float(s[i]), 3)}
                for i in np.argsort(-s)[:k] if s[i] >= 0.10]


# ----------------------------------------------------------------------------- 3. LLM
class LLM:
    """Thin wrapper over the Anthropic API. `available` is False without a key/package, and every
    caller must then use its offline fallback."""

    def __init__(self):
        self.client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                pass

    @property
    def available(self):
        return self.client is not None

    def complete(self, system: str, prompt: str, max_tokens=700) -> str:
        r = self.client.messages.create(model=LLM_MODEL, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")


CLINICIAN_SYSTEM = (
    "You are a clinical documentation assistant. Use ONLY the facts provided. Cite guideline IDs "
    "like [G3] for every recommendation. Never invent lab values, diagnoses, or medication changes. "
    "State uncertainty plainly. End with 'Decision support only - clinician review required.'")
PATIENT_SYSTEM = (
    "You write discharge guidance for patients at a 6th-grade reading level: warm, plain, no jargon, "
    "no fear-mongering, no medication changes. Use only the facts and guidelines provided.")


def generate_summary(llm: LLM, facts: dict, guidelines: list, audience="clinician") -> str:
    """Grounded generation: facts + retrieved guidelines go into the prompt; output must cite them."""
    ctx = "\n".join(f"[{g['source'].split(':')[0]}] {g['source'].split(': ', 1)[1]}: {g['text']}" for g in guidelines)
    drivers = "; ".join(f"{d['feature']}={d['value']} ({d['effect_on_risk']:+.3f})" for d in facts["drivers"])
    if llm.available:
        prompt = (f"Patient facts:\n- 30-day readmission risk: {facts['risk']:.1%} ({facts['tier']}), "
                  f"model uncertainty +/-{facts['uncertainty']:.1%}\n- Top risk drivers: {drivers}\n"
                  f"- Similar-patient segment readmission rate: {facts.get('segment_rate', float('nan')):.1%}\n\n"
                  f"Guidelines:\n{ctx}\n\n"
                  + ("Write a 5-7 line clinician handoff note with prioritized actions."
                     if audience == "clinician" else "Write a short, friendly note to the patient about what to watch for and who to call."))
        return llm.complete(CLINICIAN_SYSTEM if audience == "clinician" else PATIENT_SYSTEM, prompt)
    # ---- offline template fallback (deterministic, still grounded + cited)
    acts = "\n".join(f"  - {g['text']} [{g['source'].split(':')[0]}]" for g in guidelines)
    if audience == "clinician":
        return (f"30-day readmission risk {facts['risk']:.1%} ({facts['tier']}; +/-{facts['uncertainty']:.1%}).\n"
                f"Main drivers: {drivers}.\nSuggested actions:\n{acts}\n"
                "Decision support only - clinician review required.")
    return (f"Going home is a big step, and we want you to stay well. Your care team has noted that you may need "
            f"extra support after leaving the hospital ({facts['tier'].lower()} chance of coming back soon). "
            "Please take your medicines as listed, go to your follow-up visit, and call your clinic if you feel worse. "
            "If you have chest pain, trouble breathing, or feel very unwell, call emergency services.")
