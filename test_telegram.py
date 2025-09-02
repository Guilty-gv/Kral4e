# test_telegram.py
import os
from telegram import Bot

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
CHAT_ID = os.getenv("CHAT_ID")

print("=== DEBUG: START TEST ===")
print("TELEGRAM_TOKEN found:", bool(os.getenv("TELEGRAM_TOKEN")))
print("CHAT_ID found:", CHAT_ID)

try:
    bot.send_message(CHAT_ID, "✅ Test message from GitHub Actions!")
    print("Telegram message successfully sent!")
except Exception as e:
    print("Telegram send error:", e)

print("=== DEBUG: END TEST ===")
