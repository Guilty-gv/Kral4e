import time
import base64
import hmac
import hashlib
import requests
import os

# Земи ги од env (GitHub Actions secrets ќе бидат map-ирани тука)
API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
BASE_URL = "https://api.kucoin.com"

def sign_request(method, endpoint, body=""):
    now = int(time.time() * 1000)
    str_to_sign = str(now) + method + endpoint + body
    signature = base64.b64encode(
        hmac.new(API_SECRET.encode("utf-8"), str_to_sign.encode("utf-8"), hashlib.sha256).digest()
    )
    passphrase = base64.b64encode(
        hmac.new(API_SECRET.encode("utf-8"), API_PASSPHRASE.encode("utf-8"), hashlib.sha256).digest()
    )
    headers = {
        "KC-API-KEY": API_KEY,
        "KC-API-SIGN": signature.decode(),
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": passphrase.decode(),
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }
    return headers

def get_symbols():
    endpoint = "/api/v1/symbols"
    url = BASE_URL + endpoint
    headers = sign_request("GET", endpoint)
    response = requests.get(url, headers=headers)
    data = response.json()
    return data

def main():
    data = get_symbols()
    if "data" not in data:
        print("❌ Грешка:", data)
        return

    symbols = data["data"]
    print(f"Вкупно симболи: {len(symbols)}")

    usdt_symbols = [s for s in symbols if s["quoteCurrency"] == "USDT"]
    print(f"USDT симболи најдени: {len(usdt_symbols)}")

    for s in usdt_symbols[:20]:  # прикажи првите 20 за пример
        print(f"{s['symbol']} (base={s['baseCurrency']}, quote={s['quoteCurrency']})")

if __name__ == "__main__":
    main()
