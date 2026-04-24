from src.data_loader import build_yahoo_observed_option_frame, YahooOptionContractConfig
import pandas as pd


def main() -> None:
    config = YahooOptionContractConfig(
        ticker="SPY",
        expiration=None,
        option_type="call",
        strike=None,
        rate=0.02,
        history_period="max",
    )
    df, metadata = build_yahoo_observed_option_frame(config)
    pd.options.display.max_columns = 20
    print(metadata)
    print(df.head())
    print(len(df))


if __name__ == "__main__":
    main()
