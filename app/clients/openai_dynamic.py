import os
import time
from typing import Callable, Optional

from langchain_openai import ChatOpenAI


class DynamicChatOpenAI:
    """ChatOpenAI wrapper with dynamic API key management.

    - Provides `invoke(messages)` compatible interface.
    - Refreshes API key when expired or on auth errors.
    - Supports lazy initialization to pick up env changes.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        api_key_provider: Optional[Callable[[], str]] = None,
        refresh_key_provider: Optional[Callable[[], str]] = None,
        expires_at_provider: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key_provider = api_key_provider or (lambda: os.getenv("OPENAI_API_KEY", ""))
        self._refresh_key_provider = refresh_key_provider or _env_refresh_provider
        self._expires_at_provider = expires_at_provider or (lambda: _read_env_expiry())
        self._client: Optional[ChatOpenAI] = None
        self._api_key_cache: Optional[str] = None

    def _ensure_client(self) -> None:
        key = self._api_key_provider() or ""
        expired = _is_expired(self._expires_at_provider())
        if expired or not self._client or key != self._api_key_cache:
            self._api_key_cache = key
            # ChatOpenAI reads key from env; pass-through model/temperature
            self._client = ChatOpenAI(model=self.model, temperature=self.temperature)

    def _try_refresh_key(self) -> None:
        if self._refresh_key_provider:
            new_key = self._refresh_key_provider() or ""
            if new_key:
                # Put into env so langchain_openai can pick it up on next init
                os.environ["OPENAI_API_KEY"] = new_key
                self._api_key_cache = new_key
                # Reset client to reinitialize with new key
                self._client = None

    def invoke(self, messages):
        self._ensure_client()
        try:
            return self._client.invoke(messages)  # type: ignore[union-attr]
        except Exception as e:
            msg = str(e).lower()
            if "invalid api key" in msg or "401" in msg or "unauthorized" in msg:
                self._try_refresh_key()
                self._ensure_client()
                return self._client.invoke(messages)  # type: ignore[union-attr]
            raise


def _read_env_expiry() -> Optional[float]:
    """Read expiry timestamp (epoch seconds) from env if present."""
    val = os.getenv("OPENAI_API_KEY_EXPIRES_AT")
    if not val:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _is_expired(expires_at: Optional[float]) -> bool:
    if not expires_at:
        return False
    return time.time() >= expires_at


def _env_refresh_provider() -> str:
    """Default refresh provider: rotate to env var `OPENAI_API_KEY_ROTATE_TO` if set."""
    return os.getenv("OPENAI_API_KEY_ROTATE_TO", "")
