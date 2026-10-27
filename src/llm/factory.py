from src.llm.config import get_model_config
from src.llm.providers import OpenAIProvider, AnthropicProvider


def get_llm(model_type: str):
    config = get_model_config(model_type)

    if config.provider == "openai":
        return OpenAIProvider(config)
    elif config.provider == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"지원하지 않는 provider입니다: {config.provider}")