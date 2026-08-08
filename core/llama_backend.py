import logging
import os
from typing import Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.2


class LlamaBackend:
    """Ollama backend REST client."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

        try:
            self.timeout = int(timeout or os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT

        try:
            self.temperature = float(
                temperature
                if temperature is not None
                else os.getenv("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE)
            )
        except (TypeError, ValueError):
            self.temperature = DEFAULT_TEMPERATURE

        self._session = requests.Session()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        prompt = str(prompt).strip() if prompt else ""
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        if system_prompt:
            payload["system"] = str(system_prompt)

        url = f"{self.base_url}/api/generate"

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as error:
            raise ConnectionError(
                f"Ollama reachable error at {self.base_url}."
            ) from error
        except Exception as error:
            raise RuntimeError(f"Ollama generation failed: {error}") from error

        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        generated_text = str(data.get("response", "")).strip()

        if not generated_text:
            raise RuntimeError("Empty response received from backend.")

        return generated_text

    def is_available(self) -> bool:
        try:
            return self._session.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False