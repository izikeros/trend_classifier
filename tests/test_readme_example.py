import os

import pandas as pd
import pytest

from trend_classifier import Segmenter


@pytest.fixture(scope="module")
def apple_stock_data():
    data_file = "apple_stock_data.csv"
    if not os.path.exists(data_file):
        import yfinance as yf

        # Download data if file doesn't exist
        df = yf.download(
            "AAPL", start="2018-09-15", end="2022-09-05", interval="1d", progress=False
        )
        df.to_csv(data_file)
    else:
        # Load data from file
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    return df


def test_readme_example(apple_stock_data):
    x_in = list(range(0, len(apple_stock_data.index), 1))
    y_in = apple_stock_data["Adj Close"].tolist()
    seg = Segmenter(x_in, y_in, n=20)
    seg.calculate_segments()

    assert len(seg.segments) > 0
