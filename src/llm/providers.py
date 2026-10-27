import os

from openai import OpenAI
import anthropic

from src.llm.base import BaseLLM
from src.llm.config import ModelConfig

class OpenAIProvider(BaseLLM):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "120"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float | None = None,
        **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()


class AnthropicProvider(BaseLLM):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = anthropic.Anthropic(
            api_key=config.api_key
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        response = self.client.messages.create(
            model=self.config.model_name,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text.strip()