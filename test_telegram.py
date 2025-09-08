import os
import logging
from telegram import Bot

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logger = logging.getLogger("test_bot")
logging.basicConfig(level=logging.INFO)

# Иницијализација на Bot
bot = None
if TELEGRAM_TOKEN and CHAT_ID:
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        logger.info("Telegram Bot initialized successfully.")
    except Exception as e:
        bot = None
        logger.error("Telegram init failed: %s", e)
else:
    logger.warning("Telegram token or chat_id missing!")

def send_telegram(msg: str):
    if not bot:
        logger.warning("Bot not initialized, skipping send.")
        return
    try:
        bot.send_message(chat_id=CHAT_ID, text=msg)
        logger.info("Telegram message sent: %s", msg)
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)

# Тест порака
send_telegram("✅ Test message from GitHub workflow!")
