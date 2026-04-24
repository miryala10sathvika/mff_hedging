import yfinance as yf
ticker = yf.Ticker("SPY")
expirations = ticker.options
print("Furthest Expirations:", expirations[-5:])
if expirations:
    chain = ticker.option_chain(expirations[-1])
    calls = chain.calls
    # find highest open interest
    best_call = calls.sort_values("openInterest", ascending=False).iloc[0]
    print("Best call:", best_call["contractSymbol"])
    best_ticker = yf.Ticker(best_call["contractSymbol"])
    hist = best_ticker.history(period="max")
    print(hist.head())
    print("Length:", len(hist))
