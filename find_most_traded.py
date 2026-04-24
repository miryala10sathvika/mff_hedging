from src.data_loader import fetch_current_option_chain
import yfinance as yf
chain, u, exp = fetch_current_option_chain("SPY", expiration="2026-12-18", option_type="call")
best_sym = None
max_len = 0
for sym in chain['contractSymbol'].values:
    h = yf.Ticker(sym).history(period="max")
    if len(h) > max_len:
        max_len = len(h)
        best_sym = sym
        print(f"New best: {sym} with {max_len} days of history. Strike:", chain.loc[chain['contractSymbol'] == sym, 'strike'].iloc[0])
print(f"Winning symbol: {best_sym} with {max_len} days")
