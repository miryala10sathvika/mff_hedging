from __future__ import annotations

from src.data_loader import (
    fetch_current_option_chain,
    select_option_contract,
)

import yfinance as yf


def main() -> None:
    ticker = "SPY"
    chain, underlying, expiration = fetch_current_option_chain(ticker, expiration=None, option_type="call")
    spot = float(underlying.get("regularMarketPrice") or underlying.get("previousClose"))
    contract = select_option_contract(chain, underlying_spot=spot)
    symbol = str(contract["contractSymbol"])

    print("expiration:", expiration)
    print("contract:", symbol)
    print("spot:", spot)

    download_df = yf.download(symbol, period="max", auto_adjust=False, progress=False)
    print("\nyf.download columns:")
    print(download_df.columns)
    print(download_df.head())

    ticker_df = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    print("\nTicker.history columns:")
    print(ticker_df.columns)
    print(ticker_df.head())


if __name__ == "__main__":
    main()

