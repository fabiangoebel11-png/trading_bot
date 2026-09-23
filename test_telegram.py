"""Send one Telegram message to verify the local .env credentials."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from core.notifications.telegram_bot import TelegramBot


def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
    bot = TelegramBot()
    if not bot.enabled:
        print("Telegram test failed: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return 1

    message = "🚀 Trading Bot System-Check: Telegram Integration erfolgreich!"
    if bot.send_message(message):
        print("Telegram test succeeded: message sent.")
        return 0

    print("Telegram test failed: Telegram rejected the message or could not be reached.")
    return 1


if __name__ == "__main__":
    sys.exit(main())