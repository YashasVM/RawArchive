from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
DATASETS_DIR = DATA_DIR / "datasets"
BUNDLES_DIR = DATA_DIR / "bundles"
MODELS_DIR = DATA_DIR / "models"
APP_SECRET = os.getenv("APP_SECRET", "change-this-secret-in-production")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
STORE_RAW_UPLOADS = os.getenv("STORE_RAW_UPLOADS", "false").lower() in {"1", "true", "yes"}
DEFAULT_NOTEBOOK_URL = os.getenv(
    "NOTEBOOK_URL",
    "https://colab.research.google.com/github/your-org/ig-style-clone/blob/main/colab/train_lora.ipynb",
)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

for directory in (DATA_DIR, DATASETS_DIR, BUNDLES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
