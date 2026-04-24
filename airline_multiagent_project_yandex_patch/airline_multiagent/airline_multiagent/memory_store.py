from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from airline_multiagent.config import SETTINGS
from airline_multiagent.guards import mask_pii

DATA_DIR = Path(SETTINGS.profile_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_PATH = DATA_DIR / "passenger_profile.json"


DEFAULT_PROFILE = {
    "name": "",
    "passport": "",
    "seat_preference": "",
    "meal_preference": "",
    "home_airport": "Moscow",
    "loyalty_tier": "",
}


def _sanitize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    safe_profile = dict(DEFAULT_PROFILE)
    safe_profile.update(profile or {})

    for key in ["email", "passport"]:
        value = str(safe_profile.get(key, "") or "")
        safe_profile[key] = mask_pii(value)

    return safe_profile


def load_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        save_profile(DEFAULT_PROFILE)
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    safe_profile = _sanitize_profile(profile)

    if safe_profile != profile:
        save_profile(safe_profile)

    return safe_profile


def save_profile(profile: dict[str, Any]) -> None:
    safe_profile = _sanitize_profile(profile)
    PROFILE_PATH.write_text(
        json.dumps(safe_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_profile(key: str, value: str) -> dict[str, Any]:
    profile = load_profile()
    profile[key] = value
    save_profile(profile)
    return load_profile()
