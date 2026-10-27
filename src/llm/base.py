from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """
    모든 LLM provider가 따라야 하는 공통 인터페이스
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        텍스트 생성 함수

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 입력
            temperature: 창의성 조절
            max_tokens: 최대 토큰 수

        Returns:
            str: 생성된 응답 텍스트
        """
        pass