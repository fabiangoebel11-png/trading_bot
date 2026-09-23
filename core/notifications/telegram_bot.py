"""Small, failure-isolated Telegram Bot API client.

Notifications are observability only: a Telegram outage must never stop market
data, risk calculation, paper execution, or state recovery.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:
    import requests
except ImportError:  # pragma: no cover - optional runtime dependency
    requests = None


@dataclass(frozen=True)
class TelegramConfig:
    token: str = ""
    chat_id: str = ""
    timeout_seconds: float = 10.0
    max_retries: int = 3
    backoff_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )


class TelegramBot:
    def __init__(self, config: TelegramConfig | None = None, session=None) -> None:
        self.config = config or TelegramConfig.from_env()
        self.session = session or (requests.Session() if requests is not None else None)

    @property
    def enabled(self) -> bool:
        return bool(self.config.token and self.config.chat_id and self.session is not None)

    def send_message(self, text: str, *, disable_web_page_preview: bool = True) -> bool:
        """Send a message, returning False for disabled or exhausted delivery."""
        if not self.enabled or not text.strip():
            return False
        url = f"https://api.telegram.org/bot{self.config.token}/sendMessage"
        payload = {"chat_id": self.config.chat_id, "text": text, "disable_web_page_preview": disable_web_page_preview}
        for attempt in range(max(1, self.config.max_retries)):
            try:
                response = self.session.post(url, json=payload, timeout=self.config.timeout_seconds)
                if response.ok:
                    return True
                if response.status_code < 500 and response.status_code != 429:
                    return False
            except Exception:  # noqa: BLE001 - notifications must fail closed
                pass
            if attempt + 1 < self.config.max_retries:
                time.sleep(self.config.backoff_seconds * (2**attempt))
        return False

    def send_event(self, event: str, **fields: object) -> bool:
        lines = [f"Trading Bot | {event}"]
        lines.extend(f"{key}: {value}" for key, value in fields.items())
        return self.send_message("\n".join(lines))