import yfinance as yf
from trend_classifier import Segmenter


def test_readme_example():
    # download data from yahoo finance
    df = yf.download(
        "AAPL", start="2018-09-15", end="2022-09-05", interval="1d", progress=False
    )

    x_in = list(range(0, len(df.index.tolist()), 1))
    y_in = df["Adj Close"].tolist()

    seg = Segmenter(x_in, y_in, n=20)
    seg.calculate_segments()
