import yfinance as yf
ticker = yf.Ticker("SPY")
expirations = ticker.options
print("Expirations:", expirations[:5])
if expirations:
    chain = ticker.option_chain(expirations[0])
    calls = chain.calls
    if len(calls):
        best_call = calls.sort_values("volume", ascending=False).iloc[0]
        print("Best call:", best_call["contractSymbol"])
        best_ticker = yf.Ticker(best_call["contractSymbol"])
        hist = best_ticker.history(period="max")
        print("Length:", len(hist))
