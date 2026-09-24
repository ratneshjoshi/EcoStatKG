"""
EcoStatKG Configuration
========================
Central configuration for the EcoStatKG pipeline.
All API keys loaded from .env file.
"""

import os
import logging
import httpx
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_BASE = PROJECT_ROOT.parent          # Hallucination/
SCHEMA_PATH = PROJECT_ROOT / "schema" / "statistical_relations.json"

# Optional domain namespacing (for cross-domain runs). Default (unset) keeps the
# original environmental-sustainability paths untouched.
DOMAIN = os.getenv("ECOSTATKG_DOMAIN", "").strip()
_DOM = f"_{DOMAIN}" if DOMAIN else ""
DATA_DIR = PROJECT_ROOT / f"data{_DOM}"
CORPUS_DIR = DATA_DIR / "corpus"
TRIPLES_DIR = DATA_DIR / "triples"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = DATA_DIR / "chromadb"
ECOSTATS_DIR = DATA_DIR / "ecostats"
EVALUATION_DIR = DATA_DIR / "evaluation"
REPORTS_DIR = DATA_DIR / "reports"
SEED_TOPICS_PATH = DATA_DIR / "seed_topics.txt"
RESULTS_DIR = PROJECT_BASE / f"results{_DOM}"

LOGS_DIR = PROJECT_BASE / "logs"

# Create directories
for d in [CORPUS_DIR, TRIPLES_DIR, EMBEDDINGS_DIR, CHROMA_DIR, ECOSTATS_DIR, EVALUATION_DIR, REPORTS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Logging ===
_log_file = LOGS_DIR / f"ecostats_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),  # also print to console
    ],
)
log = logging.getLogger("ecostats")
log.info("Logging to %s", _log_file)

# === API Configuration ===
API_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
_VERIFY_SSL = os.getenv("LLM_VERIFY_SSL", "true").lower() not in ("false", "0", "no")
HTTP_CLIENT = httpx.Client(verify=_VERIFY_SSL, timeout=60.0)

# === Model Registry ===
# Generator models (for inference experiments)
# NOTE: qwen-3.6-27b is a "thinking" model — content=None, answer in `reasoning`.
#       Must pass extra_body={"chat_template_kwargs": {"enable_thinking": False}}
#       to get normal content. Other models return content directly.
#       glm-5 frequently times out — use as fallback only.
GENERATOR_MODELS = {
    "ministral-14b": "ministral-3-14b-instruct-2512",
    "mistral-small-24b": "Mistral-Small-24B-Instruct-2501-FP8-dynamic",
    "devstral-24b": "devstral-small-2-24b-instruct-2512",
    "qwen-27b": "qwen-3.6-27b",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "glm-5": "glm-5",
    "gpt-oss-120b": "gpt-oss-120b",
}

# Extraction model (for KG triple extraction — must return content directly)
EXTRACTION_MODEL = "devstral-small-2-24b-instruct-2512"

# Embedding model (for dense retrieval)
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_DIMENSIONS = 4096

# Reranker model (optional, for improved retrieval)
RERANKER_MODEL = "qwen3-reranker-8b"

# === Retrieval Parameters ===
TOP_K_DEFAULT = 5
TOP_K_ABLATION = [1, 3, 5, 7, 10]
CHUNK_SIZES_ABLATION = [512, 256, 128, 64, 32]  # token counts for granularity ablation

# === Generation Parameters ===
TEMPERATURE = 0.1  # Low temperature for factual precision
MAX_TOKENS = 256
TOP_P = 0.95

# === Evaluation Parameters ===
NUM_EVAL_SAMPLES = 500  # Size of static EcoStats test set
NUM_SEEDS = 3           # Number of seeds for statistical significance
RANDOM_SEED = 42

# === Held-Out Evaluation ===
HELD_OUT_TOPICS_PATH = ECOSTATS_DIR / "held_out_topics.json"

def load_held_out_topics() -> list[str]:
    """Load the list of held-out topics for topic-disjoint evaluation."""
    import json as _json
    if HELD_OUT_TOPICS_PATH.exists():
        with open(HELD_OUT_TOPICS_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    return []

# === Rate Limiting ===
MAX_CALLS_PER_MINUTE = 30
BATCH_SIZE = 25  # Stay under rate limit with buffer

# === System Prompts ===
GROUNDED_SYSTEM_PROMPT = """You are a factual assistant specializing in environmental statistics. Answer the question using the provided context. When the context contains specific numbers, percentages, or statistics, use those exact values. Do not fabricate numerical claims that are not supported by the context. If the context does not contain enough information, state that clearly."""

UNGROUNDED_SYSTEM_PROMPT = """You are a factual assistant specializing in environmental statistics. Answer the question as accurately as possible. When citing numbers, percentages, or statistics, be precise. Do not fabricate numerical claims you are uncertain about."""

# Legacy aliases — point to unified prompts
HPTI_SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT  # backward compat only
RAG_SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT
ZERO_SHOT_SYSTEM_PROMPT = UNGROUNDED_SYSTEM_PROMPT

TRIPLE_EXTRACTION_PROMPT = """You are a precise information extraction system. Given a text passage about {topic}, extract ALL statistical/numerical facts as structured triples.

Each triple must follow the format:
{{"subject": "<entity>", "relation": "<one of the allowed relations>", "object": "<numerical/statistical value>"}}

ALLOWED RELATIONS (use ONLY these):
{relations_list}

RULES:
1. The object (tail entity) MUST contain a specific number, percentage, rate, quantity, or date-bound statistic.
2. Do NOT extract qualitative or opinion-based statements.
3. Each triple must be self-contained — a reader should understand the fact from the triple alone.
4. If a sentence contains multiple statistics, extract one triple per statistic.
5. Prefer specific relations (HasGrowthRate, HasCapacity) over general ones (HasStatistic, HasNumericValue).

TEXT:
{text}

Extract all statistical triples as a JSON array:"""

QUESTION_GENERATION_PROMPT = """Given the following verified statistical fact, generate a natural language question that specifically targets this statistic.

Fact: The {relation} of {subject} is {object}.

Generate a question that:
1. Asks for the specific numerical value
2. Sounds natural (as a human would ask)
3. Does NOT contain the answer
4. Could plausibly have multiple similar numerical answers (to test precision)

Return ONLY the question, nothing else."""
