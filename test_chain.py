from src.data_loader import fetch_current_option_chain


def main() -> None:
    chain, u, exp = fetch_current_option_chain("SPY", None, "call")
    print("Exp:", exp, "Underlying:", float(u.get("regularMarketPrice") or u.get("previousClose")))
    print(chain[["contractSymbol", "strike", "openInterest", "volume"]].sort_values("strike").head(15))


if __name__ == "__main__":
    main()
