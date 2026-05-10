"""
Azure OpenAI Client Manager for Starlight AI-CRM Mailer.
Handles all generation and embedding tasks for the Starlight AI-CRM Mailer.

Required .env variables:
    AZURE_OPENAI_ENDPOINT             - https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY              - your Azure OpenAI key
    AZURE_OPENAI_API_VERSION          - e.g. 2024-12-01-preview
    AZURE_OPENAI_DEPLOYMENT_NAME      - your GPT-4o deployment name  (e.g. gpt-4o)
                                        alias: AZURE_OPENAI_CHAT_DEPLOYMENT
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT - your embedding deployment name
                                        (e.g. text-embedding-ada-002)
"""
import os
import io
import base64
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError, APIError, APIConnectionError

# Search for .env starting from this file's directory upward
_here = Path(__file__).parent
load_dotenv(dotenv_path=_here / ".env")

log = logging.getLogger("azure_client")


def _check_env() -> None:
    """Raise a descriptive error if required Azure credentials are missing."""
    missing = []
    if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not os.getenv("AZURE_OPENAI_API_KEY", "").strip():
        missing.append("AZURE_OPENAI_API_KEY")

    if missing:
        env_path = _here / ".env"
        raise EnvironmentError(
            f"\n\n{'='*60}\n"
            f"  MISSING required environment variables:\n"
            + "".join(f"    ✗  {v}\n" for v in missing)
            + f"\n  Create / edit the file:\n"
            f"    {env_path}\n\n"
            f"  It must contain:\n"
            f"    AZURE_OPENAI_ENDPOINT=https://openai-04.openai.azure.com/\n"
            f"    AZURE_OPENAI_API_KEY=<your key>\n"
            f"    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o\n"
            f"    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002\n"
            f"{'='*60}\n"
        )

    # Warn if endpoint looks malformed (missing https://)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if endpoint and not endpoint.startswith("https://"):
        raise EnvironmentError(
            f"AZURE_OPENAI_ENDPOINT must start with 'https://'\n"
            f"  Current value: '{endpoint}'\n"
            f"  Expected:      'https://openai-04.openai.azure.com/'"
        )


class AzureOpenAIManager:
    """
    Manages Azure OpenAI client connections for chat completions and embeddings.
    Provides retry logic with exponential backoff for rate-limit handling.
    """

    def __init__(self) -> None:
        _check_env()   # raises EnvironmentError with clear message if misconfigured

        self.endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
        self.api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Accept AZURE_OPENAI_DEPLOYMENT_NAME (primary) or legacy AZURE_OPENAI_CHAT_DEPLOYMENT
        self.chat_deployment: str = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or "gpt-4o"
        )
        self.embedding_deployment: str = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )

        log.info(
            "Azure OpenAI configured: endpoint=%s  chat=%s  embedding=%s",
            self.endpoint, self.chat_deployment, self.embedding_deployment,
        )
        self._client: Optional[AzureOpenAI] = None

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    def get_client(self) -> AzureOpenAI:
        return self.client

    def get_chat_deployment(self) -> str:
        return self.chat_deployment

    def get_embedding_deployment(self) -> str:
        return self.embedding_deployment

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
    ) -> str:
        """
        Generate a chat completion with exponential-backoff retry on rate limits.

        Args:
            messages:    OpenAI messages list, e.g. [{"role":"system","content":"..."}]
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Maximum tokens to generate.
            json_mode:   If True, force JSON output via response_format.
            max_retries: How many times to retry on RateLimitError.

        Returns:
            The generated text as a plain string.
        """
        kwargs: Dict[str, Any] = {
            "model": self.chat_deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        wait = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt == max_retries:
                    log.error("Rate limit persists after %d retries.", max_retries)
                    raise
                log.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60)
            except APIError as e:
                log.error("Azure OpenAI API error: %s", e)
                raise

        return ""  # unreachable, but satisfies type checker

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string and return its vector."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_deployment,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Embed a list of documents.

        Azure OpenAI processes up to 16 inputs per request; this method
        batches automatically and preserves order.

        Args:
            texts:      List of text strings to embed.
            batch_size: Number of texts per API call (max 16 for Azure).

        Returns:
            List of embedding vectors in the same order as input.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_deployment,
            )
            # Sort by index to guarantee order when batching
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(d.embedding for d in sorted_data)
        return all_embeddings

    def vision_completion(
        self,
        image_bytes: bytes,
        text_prompt: str,
        system_prompt: str = "",
        image_mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        max_retries: int = 4,
    ) -> str:
        """
        Send an image + text prompt to a vision-capable GPT-4o deployment.

        The image is base64-encoded and sent inline (data URI).  This works
        regardless of whether the image is publicly accessible.

        Args:
            image_bytes:   Raw bytes of the image (PNG / JPEG).
            text_prompt:   The user-side text instruction.
            system_prompt: Optional system message.
            image_mime:    MIME type, default "image/png".
            temperature:   Sampling temperature.
            max_tokens:    Max tokens in the completion.
            json_mode:     Force JSON-object output format.
            max_retries:   Retry attempts on RateLimitError.

        Returns:
            Generated text as a plain string.
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image_mime};base64,{image_b64}"

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"},
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ],
        })

        kwargs: Dict[str, Any] = {
            "model": self.chat_deployment,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        wait = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except RateLimitError:
                if attempt == max_retries:
                    raise
                log.warning(
                    "Vision rate limit (attempt %d/%d). Retrying in %ds…",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60)
            except APIError as exc:
                log.error("Azure OpenAI vision API error: %s", exc)
                raise

        return ""


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly:
#   from azure_client import azure_manager
# ---------------------------------------------------------------------------
azure_manager = AzureOpenAIManager()
