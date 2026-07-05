# ============================================================
# GuardGPT - llama_backend.py
# ============================================================
# PURPOSE:
#   Handle all communication with the Ollama LLM server.
#   This module is ONLY called after a prompt has been ALLOWED
#   by the full safety pipeline.
#
# CONFIGURATION (set these as environment variables if needed):
#   LLAMA_HOST    → Ollama server URL  (default: http://localhost:11434)
#   LLAMA_MODEL   → Model to use       (default: llama3)
#   LLAMA_TIMEOUT → Request timeout    (default: 300 seconds)
# ============================================================

import logging, os
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# Read configuration from environment variables (or use defaults)
LLAMA_HOST    = os.getenv("LLAMA_HOST",    "http://localhost:11434")
LLAMA_MODEL   = os.getenv("LLAMA_MODEL",   "llama3")
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "300"))  # 5 minutes


class LlamaBackend:
    """
    Sends approved prompts to the Ollama server and returns responses.

    Usage:
        llama = LlamaBackend()
        if llama.is_available():
            response = llama.generate("What is Python?")
    """

    def __init__(self, host=LLAMA_HOST, model=LLAMA_MODEL, timeout=LLAMA_TIMEOUT):
        self.host    = host.rstrip("/")   # remove trailing slash if present
        self.model   = model
        self.timeout = timeout

        # Build API endpoint URLs once at startup
        self._generate_url = f"{self.host}/api/generate"
        self._tags_url     = f"{self.host}/api/tags"

        logger.info("LlamaBackend ready: model=%s  host=%s", self.model, self.host)

    def is_available(self) -> bool:
        """
        Check if the Ollama server is running and reachable.
        Uses a quick 5-second timeout to avoid hanging.
        Returns True if server responds with HTTP 200, False otherwise.
        """
        try:
            response = requests.get(self._tags_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False  # server not running or unreachable

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to Ollama and return the model's response text.

        Args:
            prompt        : The user's approved message to send to the LLM.
            system_prompt : Optional instruction that guides the LLM's behaviour.

        Returns:
            The model's response as a plain string.

        Raises:
            ConnectionError : If the Ollama server is not running.
            RuntimeError    : If the server returns an error or unexpected response.
        """
        # Build the request payload
        payload = {
            "model" : self.model,
            "prompt": prompt,
            "stream": False,   # wait for full response, don't stream
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        logger.debug("Sending prompt to Llama (%d chars)", len(prompt))

        # Send the request to Ollama
        try:
            resp = requests.post(
                self._generate_url,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()  # raise error for HTTP 4xx/5xx responses

        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}.\n"
                "Make sure Ollama is running: open a new terminal and run 'ollama serve'"
            )
        except requests.HTTPError:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama did not respond within {self.timeout} seconds. "
                "Try again or use a smaller model."
            )

        # Parse the response
        data = resp.json()
        if "response" not in data:
            raise RuntimeError(
                f"Unexpected response format from Ollama: {str(data)[:200]}"
            )

        text = data["response"].strip()
        logger.debug("Received response from Llama (%d chars)", len(text))
        return text