from trend_classifier.configuration import Config
from trend_classifier.segmentation import Segmenter


class TestCalculateSegments:
    def setup_method(self):
        self.seg = Segmenter(config=Config())

    def test_calculate_segments__lambda_shape(self):
        self.seg.x = list(range(0, 200))
        self.seg.y = list(range(0, 100)) + list(range(100, 0, -1))
        segments = self.seg.calculate_segments()
        # FIXME: KS: 2022-09-01: Expecting to have 2 segments,
        #  but currently getting 3 with default config
        assert len(segments) == 3

    def test_calculate_segments__v_shape(self):
        self.seg.x = list(range(0, 200))
        self.seg.y = list(range(100, 0, -1)) + list(range(0, 100))
        segments = self.seg.calculate_segments()
        assert len(segments) == 2

    def test_calculate_segments__line_up(self):
        self.seg.x = list(range(0, 200))
        self.seg.y = list(range(0, 200))
        segments = self.seg.calculate_segments()
        # FIXME: KS: 2022-09-02: Expecting to have 1 segment, but currently getting 3
        assert len(segments) == 3

    def test_calculate_segments__line_down(self):
        self.seg.x = list(range(0, 200))
        self.seg.y = list(range(200, 0, -1))
        segments = self.seg.calculate_segments()
        assert len(segments) == 1