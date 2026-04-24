from src.data_loader import fetch_current_option_chain


def main() -> None:
    chain, u, exp = fetch_current_option_chain("SPY", None, "call")
    non_zero = chain[chain["openInterest"].fillna(0) >= 1]
    print("Expiration:", exp)
    print("Underlying:", float(u.get("regularMarketPrice") or u.get("previousClose")))
    print("Non-zero open interest:", len(non_zero))
    if len(non_zero) > 0:
        print(non_zero[["contractSymbol", "strike", "openInterest", "volume"]].sort_values("strike"))


if __name__ == "__main__":
    main()
