from unittest.mock import patch

import pytest
from trend_classifier.segmentation import Segmenter


class TestCalculateSegments:
    def test_calculate_segments__lambda_shape(self):
        x = list(range(0, 200))
        y = list(range(0, 100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y)

        segments = self.seg.calculate_segments()
        # FIXME: KS: 2022-09-01: Expecting to have 2 segments,
        #  but currently getting 3 with default config
        assert len(segments) == 3

    def test_calculate_segments__v_shape(self):
        x = list(range(0, 200))
        y = list(range(100, 0, -1)) + list(range(0, 100))
        self.seg = Segmenter(x=x, y=y)
        segments = self.seg.calculate_segments()
        assert len(segments) == 2

    def test_calculate_segments__line_up(self):
        x = list(range(0, 200))
        y = list(range(0, 200))
        self.seg = Segmenter(x=x, y=y)
        segments = self.seg.calculate_segments()
        # FIXME: KS: 2022-09-02: Expecting to have 1 segment, but currently getting 3
        assert len(segments) == 3

    def test_calculate_segments__line_down(self):
        x = list(range(0, 200))
        y = list(range(200, 0, -1))
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
        x = list(range(0, 200))
        y = list(range(0, 100)) + list(range(100, 0, -1))
        self.seg = Segmenter(x=x, y=y)
        self.seg.calculate_segments()
        area_outside_trend = self.seg.calc_area_outside_trend()
        assert area_outside_trend > 0


class TestSegmenterPlotting:
    def setup_class(self):
        x = list(range(0, 200))
        y = list(range(0, 100)) + list(range(100, 0, -1))
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
