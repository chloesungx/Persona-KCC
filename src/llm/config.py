import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    checkpoint_path: str | None = None


DEFAULT_MODEL_CONFIGS = {
    "claude-sonnet-4-6": ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=None,
    ),
    "gpt-5.4-nano": ModelConfig(
        provider="openai",
        model_name="gpt-5.4-nano",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ),
}

def get_model_config(model_type: str) -> ModelConfig:
    model_type = model_type.lower()

    if model_type not in DEFAULT_MODEL_CONFIGS:
        raise ValueError(
            f"지원하지 않는 모델 타입입니다: {model_type}. "
            f"지원 목록: {list(DEFAULT_MODEL_CONFIGS.keys())}"
        )

    return DEFAULT_MODEL_CONFIGS[model_type]