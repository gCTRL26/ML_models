from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "yandex")
    model_name: str = os.getenv("MODEL_NAME", "")
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0"))

    model_api_key: str = os.getenv("MODEL_API_KEY", "")
    model_base_url: str = os.getenv("MODEL_BASE_URL", "")
    yandex_cloud_folder: str = os.getenv("YANDEX_CLOUD_FOLDER", "")

    app_name: str = os.getenv("APP_NAME", "airline-multi-agent")
    profile_dir: str = os.getenv("PROFILE_DIR", "./data")
    default_thread_id: str = os.getenv("DEFAULT_THREAD_ID", "demo-thread")


SETTINGS = Settings()
