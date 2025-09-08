# test_telegram_bot.py
import os
import logging
from telegram import Bot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_test")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # за група треба да почнува со '-'

if not TELEGRAM_TOKEN or not CHAT_ID:
    logger.error("TELEGRAM_TOKEN или CHAT_ID не се поставени!")
    exit(1)

try:
    bot = Bot(token=TELEGRAM_TOKEN)
    msg = "✅ Test message from Hybrid Bot"
    bot.send_message(chat_id=CHAT_ID, text=msg)
    logger.info("Test message successfully sent!")
except Exception as e:
    logger.exception("Failed to send test message: %s", e)
