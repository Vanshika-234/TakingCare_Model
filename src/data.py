"""Data layer: loads the UCI 'Diabetes 130-US Hospitals' dataset (1999-2008) if present,
otherwise generates a synthetic cohort with the same schema so the project runs anywhere.

Target: readmitted_30d = 1 if the patient was readmitted within 30 days of discharge.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import REAL_DATA, SEED

NUM_COLS = ["age_num", "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
            "number_diagnoses", "prior_visits_total", "meds_per_day"]
CAT_COLS = ["race", "gender", "admission_type", "discharge_group", "A1Cresult",
            "insulin", "change", "diabetesMed", "diag_group"]
TARGET = "readmitted_30d"

AGE_BUCKETS = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
DIAG_CODES = {"circulatory": ["428", "410", "414"], "respiratory": ["486", "491", "518"],
              "diabetes": ["250.83", "250.13", "250.6"], "digestive": ["560", "578", "574"],
              "injury": ["820", "812", "959"], "musculoskeletal": ["715", "722"],
              "genitourinary": ["599", "584"], "neoplasms": ["153", "197"], "other": ["780", "786"]}


def icd9_group(code) -> str:
    """Map an ICD-9 code to a coarse diagnosis group (as in Strack et al., 2014)."""
    c = str(code)
    if c.startswith(("V", "E")) or c in ("?", "nan"):
        return "other"
    try:
        v = float(c)
    except ValueError:
        return "other"
    if c.startswith("250"): return "diabetes"
    if 390 <= v <= 459 or v == 785: return "circulatory"
    if 460 <= v <= 519 or v == 786: return "respiratory"
    if 520 <= v <= 579 or v == 787: return "digestive"
    if 800 <= v <= 999: return "injury"
    if 710 <= v <= 739: return "musculoskeletal"
    if 580 <= v <= 629 or v == 788: return "genitourinary"
    if 140 <= v <= 239: return "neoplasms"
    return "other"


def synthetic_cohort(n=20000, seed=SEED) -> pd.DataFrame:
    """Synthetic cohort with the UCI schema; readmission risk is driven by clinically plausible
    factors (prior inpatient stays dominate, as in the literature), plus noise and interactions."""
    r = np.random.default_rng(seed)
    age = r.choice(AGE_BUCKETS[1:], n, p=[.01, .02, .04, .09, .17, .25, .27, .13, .02])
    age_mid = np.array([int(a[1:3].strip("-")) + 5 for a in age])
    inpatient = r.poisson(np.clip(0.25 + (age_mid - 40) * 0.01, 0.05, 1.2))
    emergency = r.poisson(0.2 + 0.1 * (inpatient > 0))
    outpatient = r.poisson(0.35, n)
    tih = np.clip(r.gamma(2.2, 2.0, n).round() + 1, 1, 14).astype(int)
    labs = np.clip(r.normal(30 + 2.5 * tih, 14), 1, 130).astype(int)
    procs = r.poisson(1.2, n).clip(0, 6)
    meds = np.clip(r.normal(10 + 0.9 * tih, 5), 1, 60).astype(int)
    diags = np.clip(r.normal(6.5 + 0.2 * inpatient, 1.8), 1, 16).astype(int)
    groups = r.choice(list(DIAG_CODES), n, p=[.30, .12, .10, .10, .07, .06, .06, .04, .15])
    diag_1 = np.array([r.choice(DIAG_CODES[g]) for g in groups])
    a1c = r.choice(["None", "Norm", ">7", ">8"], n, p=[.82, .05, .06, .07])
    insulin = r.choice(["No", "Steady", "Up", "Down"], n, p=[.45, .30, .12, .13])
    change = r.choice(["No", "Ch"], n, p=[.53, .47])
    dmed = r.choice(["Yes", "No"], n, p=[.77, .23])
    adm = r.choice([1, 2, 3, 5, 6], n, p=[.53, .18, .19, .05, .05])
    dis = r.choice([1, 3, 6, 22, 11, 2], n, p=[.58, .13, .12, .06, .05, .06])
    race = r.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], n,
                    p=[.75, .19, .02, .01, .01, .02])
    gender = r.choice(["Female", "Male"], n, p=[.54, .46])

    z = (-3.15 + 0.55 * np.log1p(inpatient) * 2 + 0.25 * np.log1p(emergency) * 2
         + 0.12 * np.log1p(outpatient) + 0.035 * (tih - 4) + 0.03 * (diags - 7)
         + 0.012 * (age_mid - 65) + 0.008 * (meds - 15)
         + 0.35 * np.isin(dis, [3, 6, 22]) + 0.25 * (groups == "circulatory")
         + 0.20 * (groups == "diabetes") + 0.15 * (a1c == ">8") - 0.1 * (dmed == "No")
         + 0.30 * (inpatient > 1) * (diags > 8) + r.normal(0, 0.55, n))
    y = r.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame({
        "race": race, "gender": gender, "age": age, "admission_type_id": adm,
        "discharge_disposition_id": dis, "time_in_hospital": tih, "num_lab_procedures": labs,
        "num_procedures": procs, "num_medications": meds, "number_outpatient": outpatient,
        "number_emergency": emergency, "number_inpatient": inpatient, "number_diagnoses": diags,
        "A1Cresult": a1c, "insulin": insulin, "change": change, "diabetesMed": dmed,
        "diag_1": diag_1, "readmitted": np.where(y, "<30", "NO")})
    # inject missingness like the real data
    df.loc[r.random(n) < 0.03, "A1Cresult"] = np.nan
    return df


def load_raw() -> tuple[pd.DataFrame, str]:
    if REAL_DATA.exists():
        return pd.read_csv(REAL_DATA, na_values=["?"]), "UCI Diabetes 130-US hospitals (real)"
    return synthetic_cohort(), "synthetic cohort (UCI schema) - place diabetic_data.csv in data/ for real data"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_num"] = d["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    d["admission_type"] = d["admission_type_id"].map(
        {1: "emergency", 2: "urgent", 3: "elective", 4: "newborn", 7: "trauma"}).fillna("other")
    d["discharge_group"] = d["discharge_disposition_id"].map(
        lambda x: "home" if x in (1, 6, 8) else "facility" if x in (2, 3, 4, 5, 22, 23, 24, 30)
        else "expired_hospice" if x in (11, 13, 14, 19, 20, 21) else "other")
    d["diag_group"] = d["diag_1"].map(icd9_group)
    d["prior_visits_total"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]
    d["meds_per_day"] = d["num_medications"] / d["time_in_hospital"].clip(lower=1)
    d[TARGET] = (d["readmitted"] == "<30").astype(int)
    # real data: drop patients who died / went to hospice (cannot be readmitted)
    d = d[d["discharge_group"] != "expired_hospice"]
    return d.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def get_splits():
    """Returns dict with raw frames and split indices (70/15/15, stratified)."""
    raw, source = load_raw()
    df = engineer(raw)
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=df[TARGET], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=df.loc[tmp, TARGET], random_state=SEED)
    return {"df": df, "train": tr, "val": va, "test": te, "source": source}
