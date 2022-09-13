from unittest.mock import patch

import pytest
from trend_classifier.configuration import CONFIG_REL_SLOPE_ONLY
from trend_classifier.segmentation import Segmenter


class TestCalculateSegments:
    def test_calculate_segments__lambda_shape(self):
        x = list(range(0, 200))
        y = list(range(0, 100)) + list(range(100, 0, -1))  # noqa: FKA01
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)

        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 2

    def test_calculate_segments__v_shape(self):
        x = list(range(0, 200))
        y = list(range(100, 0, -1)) + list(range(0, 100))  # noqa: FKA01
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 2

    def test_calculate_segments__line_up(self):
        x = list(range(0, 200))
        y = list(range(0, 200))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 1

    def test_calculate_segments__line_down(self):
        x = list(range(0, 200))  # noqa: FKA01
        y = list(range(200, 0, -1))  # noqa: FKA01
        self.seg = Segmenter(x=x, y=y)
        segments = self.seg.calculate_segments()
        assert len(segments) == 1

    @pytest.mark.skip(reason="Not implemented yet")
    def test_calculate_segments__segments_not_overlap(self):
        """Check if segments are not overlapping."""
        # TODO: KS: 2022-09-07: Consider what is the better option:
        #  - to have segments not overlapping
        #  - to have segments overlapping?
        assert True

    def test_calc_area_outside_trend(self):
        x = list(range(0, 200))  # noqa: FKA01
        y = list(range(0, 100)) + list(range(100, 0, -1))  # noqa: FKA01
        self.seg = Segmenter(x=x, y=y)
        self.seg.calculate_segments()
        area_outside_trend = self.seg.calc_area_outside_trend()
        assert area_outside_trend > 0

    def test__set_offset(self):
        self.seg = Segmenter(x=[], y=[])
        offset = self.seg._set_offset(n=100, overlap_ratio=0.2)
        assert offset == 20

    def test__set_offset__adjust_to_one(self):
        self.seg = Segmenter(x=[], y=[])
        offset = self.seg._set_offset(n=10, overlap_ratio=0.001)
        assert offset == 1


class TestSegmenterPlotting:
    def setup_class(self):
        x = list(range(0, 200))  # noqa: FKA01
        y = list(range(0, 100)) + list(range(100, 0, -1))  # noqa: FKA01
        self.seg = Segmenter(x=x, y=y)
        self.seg.calculate_segments()

    @patch("matplotlib.pyplot.show")
    def test_plot_segment(self, mock_show):
        self.seg.plot_segment(0)

    @patch("matplotlib.pyplot.show")
    def test_plot_segments(self, mock_show):
        self.seg.plot_segments()

    @patch("matplotlib.pyplot.show")
    def test_plot_detrended_signal(self, mock_show):
        self.seg.plot_detrended_signal()

    @patch("matplotlib.pyplot.show")
    def test_plot_segment_with_trendlines_no_context(self, mock_show):
        self.seg.plot_segment_with_trendlines_no_context(0)


@pytest.mark.skip(reason="Use for manual testing")
def test_real_data():
    import yfinance as yf
    from trend_classifier import Segmenter

    df = yf.download(
        "AAPL", start="2018-09-15", end="2022-09-06", interval="1d", progress=False
    )

    seg = Segmenter(df=df, column="Adj Close", n=20)
    seg.calculate_segments()
