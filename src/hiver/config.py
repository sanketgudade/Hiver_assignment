"""Central config: paths, brand, model names, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw" / "twcs" / "twcs.csv"
APPLE_THREADS = DATA / "apple_threads.parquet"
APPLE_SMALL = DATA / "apple_small.parquet"
GOLDEN_SET = DATA / "golden_set.csv"
EVAL_RESULTS = ROOT / "eval" / "results"

# Brand
BRAND = "AppleSupport"

# LLM
GROQ_MODEL = "openai/gpt-oss-120b"
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 400

# Pipeline
CONFIDENCE_THRESHOLD = 0.75
AUTO_HANDLE_INTENTS = {"order_status", "general_inquiry"}
ESCALATION_KEYWORDS = [
    "refund", "charge", "fraud", "legal", "lawyer", "sue",
    "cancel my account", "delete my account", "locked out",
    "hacked", "unauthorized",
]

# Sampling
SMALL_SAMPLE_N = 20000
GOLDEN_N = 200
RANDOM_SEED = 42