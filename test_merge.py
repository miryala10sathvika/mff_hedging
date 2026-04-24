from src.experiments import YahooOptionExperimentConfig
from src.data_loader import build_yahoo_observed_option_frame, YahooOptionContractConfig
config = YahooOptionContractConfig(
    ticker="SPY",
    expiration=None,
    option_type="call",
    strike=None,
    rate=0.02,
    history_period="max"
)
df, metadata = build_yahoo_observed_option_frame(config)
import pandas as pd
pd.options.display.max_columns = 20
print(df.head())
print(len(df))
