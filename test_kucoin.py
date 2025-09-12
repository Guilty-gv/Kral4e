name: Test KuCoin API

on:
  workflow_dispatch:  # ќе можеш рачно да го стартуваш од GitHub Actions таб

jobs:
  test-kucoin:
    runs-on: ubuntu-latest
    env:
      KUCOIN_API_KEY: ${{ secrets.KUCOIN_API_KEY }}
      KUCOIN_API_SECRET: ${{ secrets.KUCOIN_API_SECRET }}
      KUCOIN_API_PASSPHRASE: ${{ secrets.KUCOIN_API_PASSPHRASE }}

    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install requests

      - name: Run KuCoin Test Script
        run: python test_kucoin.py
