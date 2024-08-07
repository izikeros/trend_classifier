from unittest.mock import patch

import pytest
from trend_classifier.configuration import CONFIG_REL_SLOPE_ONLY
from trend_classifier.segmentation import Segmenter


class TestCalculateSegments:
    """Test suite for the calculate_segments method of the Segmenter class."""

    def test_calculate_segments__lambda_shape(self):
        """Test segmentation of a lambda-shaped trend.

        Expected: 2 segments.
        """
        x = list(range(200))
        y = list(range(100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 2

    def test_calculate_segments__v_shape(self):
        """Test segmentation of a V-shaped trend.

        Expected: 2 segments.
        """
        x = list(range(200))
        y = list(range(100, 0, -1)) + list(range(100))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 2

    def test_calculate_segments__line_up(self):
        """Test segmentation of an upward linear trend.

        Expected: 1 segment.
        """
        x = list(range(200))
        y = list(range(200))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 1

    def test_calculate_segments__line_down(self):
        """Test segmentation of a downward linear trend.

        Expected: 1 segment.
        """
        x = list(range(200))
        y = list(range(200, 0, -1))
        self.seg = Segmenter(x=x, y=y)
        segments = self.seg.calculate_segments()
        assert len(segments) == 1

    def test_calculate_segments__m_shape(self):
        """Test segmentation of an M-shaped signal.

        Expected: 3 segments.
        """
        x = list(range(300))
        y = list(range(100)) + list(range(100, 0, -1)) + list(range(100))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 3

    def test_calculate_segments__w_shape(self):
        """Test segmentation of a W-shaped signal.

        Expected: 3 segments.
        """
        x = list(range(300))
        y = list(range(100, 0, -1)) + list(range(100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 3

    def test_calculate_segments__u_shape(self):
        """Test segmentation of a U-shaped signal.

        Expected: 2 segments.
        """
        x = list(range(200))
        y = list(range(100, 0, -1)) + list(range(100))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 2

    def test_calculate_segments__zigzag(self):
        """Test segmentation of a zigzag-shaped signal.

        Expected: 3 segments.
        """
        x = list(range(300))
        y = list(range(100)) + list(range(100, 0, -1)) + list(range(100))
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 3

    def test_calculate_segments__sine(self):
        """Test segmentation of a sine-shaped signal.

        Expected: 3 segment.
        """
        import numpy as np

        x = np.linspace(0, 2 * np.pi, 200).tolist()
        y = np.sin(x).tolist()
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 3

    def test_calculate_segments__sine_halfs(self):
        """Test segmentation of a sine-shaped signal with half's.

        Expected: 4 segments.
        """
        import numpy as np

        x = np.linspace(0, 2 * np.pi, 200).tolist()
        y = np.abs(np.sin(x)).tolist()
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 4

    # saw-shaped signal, 10 periods with amplitude from 1
    # in the beginning to 0 in the end (gradually suppressing the signal)
    @pytest.mark.skip(reason="Solution not working correctly")
    def test_calculate_segments__saw_10(self):
        """Test segmentation of a saw-shaped signal.

        Expected: 1 segment.
        """
        x = list(range(200))
        y = (list(range(10)) + list(range(10, 0, -1))) * 10
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 20

    @pytest.mark.skip(reason="Solution not working correctly")
    def test_calculate_segments__saw_25(self):
        """Test segmentation of a saw-shaped signal.

        Expected: 1 segment.
        """
        x = list(range(200))
        y = (list(range(25)) + list(range(25, 0, -1))) * 4
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 8

    def test_calculate_segments__saw_50(self):
        """Test segmentation of a saw-shaped signal.

        Expected: 1 segment.
        """
        x = list(range(200))
        y = (list(range(50)) + list(range(50, 0, -1))) * 2
        self.seg = Segmenter(x=x, y=y, config=CONFIG_REL_SLOPE_ONLY)
        segments = self.seg.calculate_segments()
        n_segments = len(segments)
        assert n_segments == 4

    @pytest.mark.skip(reason="Not implemented yet")
    def test_calculate_segments__segments_not_overlap(self):
        """Check if segments are not overlapping."""
        # TODO: KS: 2022-09-07: Consider what is the better option:
        #  - to have segments not overlapping
        #  - to have segments overlapping?
        assert True

    def test_calc_area_outside_trend(self):
        """Test calculation of area outside the trend."""
        x = list(range(200))
        y = list(range(100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y)
        self.seg.calculate_segments()
        area_outside_trend = self.seg.calc_area_outside_trend()
        assert area_outside_trend > 0

    def test__set_offset(self):
        """Test setting of offset based on overlap ratio."""
        self.seg = Segmenter(x=[], y=[])
        offset = self.seg._set_offset(n=100, overlap_ratio=0.2)
        assert offset == 20

    def test__set_offset__adjust_to_one(self):
        """Test adjustment of offset to 1 for small overlap ratios."""
        self.seg = Segmenter(x=[], y=[])
        offset = self.seg._set_offset(n=10, overlap_ratio=0.001)
        assert offset == 1


class TestSegmenterPlotting:
    """Test suite for the plotting methods of the Segmenter class."""

    def setup_class(self):
        """Set up the Segmenter instance for plotting tests."""
        x = list(range(200))
        y = list(range(100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y)
        self.seg.calculate_segments()

    @patch("matplotlib.pyplot.show")
    def test_plot_segment(self, mock_show):
        """Test plotting of a single segment."""
        self.seg.plot_segment(0)

    @patch("matplotlib.pyplot.show")
    def test_plot_segments(self, mock_show):
        """Test plotting of all segments."""
        self.seg.plot_segments()

    @patch("matplotlib.pyplot.show")
    def test_plot_detrended_signal(self, mock_show):
        """Test plotting of the detrended signal."""
        self.seg.plot_detrended_signal()

    @patch("matplotlib.pyplot.show")
    def test_plot_segment_with_trendlines_no_context(self, mock_show):
        """Test plotting of a segment with trendlines and no context."""
        self.seg.plot_segment_with_trendlines_no_context(0)


@pytest.mark.skip(reason="Use for manual testing")
def test_real_data():
    """Test the Segmenter with real stock market data (manual test)."""
    import yfinance as yf
    from trend_classifier import Segmenter

    df = yf.download(
        "AAPL", start="2018-09-15", end="2022-09-06", interval="1d", progress=False
    )
    seg = Segmenter(df=df, column="Adj Close", n=20)
    seg.calculate_segments()
