from src.data_loader import fetch_current_option_chain
chain, u, exp = fetch_current_option_chain("SPY", None, "call")
non_zero = chain[chain['openInterest'].fillna(0) >= 1]
print("Non-zero open interest:", len(non_zero))
if len(non_zero) > 0:
    print(non_zero[['contractSymbol','strike', 'openInterest', 'volume']].sort_values('strike'))
