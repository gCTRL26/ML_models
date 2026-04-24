from __future__ import annotations

from langchain_openai import ChatOpenAI

from airline_multiagent.config import SETTINGS


def build_chat_llm(temperature: float | None = None) -> ChatOpenAI:
    temp = SETTINGS.temperature if temperature is None else temperature

    kwargs: dict = {
        "model": SETTINGS.model_name,
        "temperature": temp,
    }

    if SETTINGS.model_api_key:
        kwargs["api_key"] = SETTINGS.model_api_key
    if SETTINGS.model_base_url:
        kwargs["base_url"] = SETTINGS.model_base_url
    if SETTINGS.yandex_cloud_folder:
        kwargs["default_headers"] = {"x-folder-id": SETTINGS.yandex_cloud_folder}

    return ChatOpenAI(**kwargs)
