"""Central configuration for CareCompass."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
REAL_DATA = DATA_DIR / "diabetic_data.csv"   # UCI "Diabetes 130-US hospitals" (optional)
SEED = 42

# LLM settings (used by Generative AI + Agentic AI layers). Set ANTHROPIC_API_KEY to enable.
LLM_MODEL = os.environ.get("CARECOMPASS_MODEL", "claude-sonnet-5-5")
MAX_AGENT_STEPS = 8

# Risk tiers on the calibrated 30-day readmission probability
TIER_HIGH, TIER_MED = 0.20, 0.10
