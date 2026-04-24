from src.data_loader import fetch_current_option_chain, select_option_contract
import yfinance as yf
ticker = yf.Ticker("SPY")
expirations = ticker.options
# Try each expiration to find one with good history
for exp in expirations:
    chain, u, _ = fetch_current_option_chain("SPY", expiration=exp, option_type="call")
    spot = float(u.get("regularMarketPrice") or u.get("previousClose"))
    try:
        contract = select_option_contract(chain, underlying_spot=spot, max_strike_distance_pct=0.05, min_open_interest=100)
    except ValueError:
        continue
    symbol = str(contract["contractSymbol"])
    hist = yf.Ticker(symbol).history(period="max")
    if len(hist) > 100:
        print(f"Found good option! Exp: {exp}, Symbol: {symbol}, History length: {len(hist)}")
        break
