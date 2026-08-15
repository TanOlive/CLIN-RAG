import os
import yaml
from pathlib import Path

# Resolve PROJECT_ROOT based on this file's location (src/config.py -> parent.parent is project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the declarative YAML configuration
config_path = PROJECT_ROOT / "config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found at: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / _cfg["paths"]["data_dir"]
TOKEN_PATH = PROJECT_ROOT / _cfg["paths"]["token_path"]
REPORTS_CSV = PROJECT_ROOT / _cfg["paths"]["reports_csv"]
PROJECTIONS_CSV = PROJECT_ROOT / _cfg["paths"]["projections_csv"]
INDEX_PATH = PROJECT_ROOT / _cfg["paths"]["index_path"]
MAPPING_PATH = PROJECT_ROOT / _cfg["paths"]["mapping_path"]
IMAGES_DIR = PROJECT_ROOT / _cfg["paths"]["images_dir"]
ENTITIES_PATH = PROJECT_ROOT / _cfg["paths"]["entities_path"]

TEST_METADATA_PATH = PROJECT_ROOT / _cfg["paths"]["test_metadata_path"]
TEST_IMAGES_DIR = PROJECT_ROOT / _cfg["paths"]["test_images_dir"]
EVALUATION_RUNS_DIR = PROJECT_ROOT / _cfg["paths"]["evaluation_runs_dir"]

TEMP_UPLOAD_DIR = PROJECT_ROOT / _cfg["paths"]["temp_upload_dir"]

# ---------------------------------------------------------------------------
# Model Identifiers
# ---------------------------------------------------------------------------

GENERATOR_MODEL_ID = _cfg["models"]["generator_id"]
ENCODER_MODEL_ID = _cfg["models"]["encoder_id"]
BERTSCORE_MODEL_ID = _cfg["models"]["bertscore_id"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

RETRIEVAL_K_CASES = _cfg["hyperparameters"]["retrieval_k_cases"]
GENERATION_MAX_NEW_TOKENS = _cfg["hyperparameters"]["generation_max_new_tokens"]
GENERATION_DO_SAMPLE = _cfg["hyperparameters"]["generation_do_sample"]
GENERATION_TEMPERATURE = _cfg["hyperparameters"]["generation_temperature"]
GENERATION_TOP_P = _cfg["hyperparameters"]["generation_top_p"]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

BERTSCORE_MODEL_MAX_LENGTH = _cfg["evaluation"]["bertscore_model_max_length"]
DEBUG_SAMPLE_LIMIT = _cfg["evaluation"]["debug_sample_limit"]

# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------

ENCODE_BATCH_SIZE = _cfg["data_processing"]["encode_batch_size"]
ENTITIES_TOP_K = _cfg["data_processing"]["entities_top_k"]
ENTITIES_STOP_WORDS = set(_cfg["data_processing"]["entities_stop_words"])
TEST_SPLIT_SAMPLES_PER_CATEGORY = _cfg["data_processing"]["test_split_samples_per_category"]
TEST_SPLIT_RANDOM_SEED = _cfg["data_processing"]["test_split_random_seed"]
TEST_SPLIT_CATEGORIES = _cfg["data_processing"]["test_split_categories"]
TEST_SPLIT_PATHO_KEYWORDS = _cfg["data_processing"]["test_split_patho_keywords"]

# ---------------------------------------------------------------------------
# UI / XAI Configurations
# ---------------------------------------------------------------------------

DEFAULT_HEATMAP_THRESHOLD = _cfg["ui"]["default_heatmap_threshold"]
XAI_BASE_ALPHA = _cfg["ui"]["xai_base_alpha"]
XAI_SMOOTHING_KERNEL_RATIO = _cfg["ui"]["xai_smoothing_kernel_ratio"]
XAI_PERCENTILE_CLIP = _cfg["ui"]["xai_percentile_clip"]
