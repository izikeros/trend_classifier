"""Tests for the detector module."""

import numpy as np
import pytest

from trend_classifier.configuration import Config
from trend_classifier.detectors import (
    BaseDetector,
    BottomUpDetector,
    DetectionResult,
    SlidingWindowDetector,
    get_detector,
    list_detectors,
)
from trend_classifier.segment import SegmentList


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        segments = SegmentList()
        result = DetectionResult(segments=segments)
        assert result.segments == segments
        assert result.breakpoints == []
        assert result.metadata == {}

    def test_with_breakpoints(self):
        """Test creation with breakpoints."""
        segments = SegmentList()
        result = DetectionResult(
            segments=segments,
            breakpoints=[10, 50, 100],
            metadata={"algorithm": "test"},
        )
        assert result.breakpoints == [10, 50, 100]
        assert result.metadata["algorithm"] == "test"

    def test_len(self):
        """Test __len__ method."""
        segments = SegmentList()
        result = DetectionResult(segments=segments)
        assert len(result) == 0


class TestSlidingWindowDetector:
    """Tests for SlidingWindowDetector."""

    def setup_method(self):
        """Set up test data."""
        self.x = np.arange(200, dtype=np.float64)
        self.y = np.concatenate(
            [
                np.linspace(0, 50, 100),
                np.linspace(50, 20, 100),
            ]
        )

    def test_basic_detection(self):
        """Test basic segment detection."""
        detector = SlidingWindowDetector(n=30, alpha=2.0, beta=2.0)
        result = detector.fit_detect(self.x, self.y)

        assert isinstance(result, DetectionResult)
        assert len(result.segments) >= 1
        assert result.metadata["algorithm"] == "sliding_window"

    def test_from_config(self):
        """Test creation from Config object."""
        config = Config(N=40, overlap_ratio=0.25, alpha=1.5, beta=1.5)
        detector = SlidingWindowDetector.from_config(config)

        assert detector.n == 40
        assert detector.overlap_ratio == 0.25
        assert detector.alpha == 1.5
        assert detector.beta == 1.5

    def test_fit_detect_chain(self):
        """Test fit().detect() chain."""
        detector = SlidingWindowDetector(n=30)
        detector.fit(self.x, self.y)
        result = detector.detect()

        assert isinstance(result, DetectionResult)
        assert len(result.segments) >= 1

    def test_data_too_short(self):
        """Test error when data is shorter than window."""
        detector = SlidingWindowDetector(n=100)
        with pytest.raises(ValueError, match="Data length"):
            detector.fit(self.x[:50], self.y[:50])

    def test_detect_before_fit_raises(self):
        """Test that detect() before fit() raises error."""
        detector = SlidingWindowDetector(n=30)
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.detect()

    def test_progress_callback(self):
        """Test progress callback is called."""
        calls = []

        def callback(current, total):
            calls.append((current, total))

        detector = SlidingWindowDetector(n=30)
        detector.fit(self.x, self.y).detect(progress_callback=callback)

        assert len(calls) > 0


class TestBottomUpDetector:
    """Tests for BottomUpDetector."""

    def setup_method(self):
        """Set up test data."""
        self.x = np.arange(200, dtype=np.float64)
        self.y = np.concatenate(
            [
                np.linspace(0, 50, 100),
                np.linspace(50, 20, 100),
            ]
        )

    def test_basic_detection(self):
        """Test basic segment detection."""
        detector = BottomUpDetector(max_segments=5)
        result = detector.fit_detect(self.x, self.y)

        assert isinstance(result, DetectionResult)
        assert len(result.segments) <= 5
        assert result.metadata["algorithm"] == "bottom_up"

    def test_max_segments_respected(self):
        """Test that max_segments limit is respected."""
        detector = BottomUpDetector(max_segments=3)
        result = detector.fit_detect(self.x, self.y)

        assert len(result.segments) <= 3

    def test_segments_cover_data(self):
        """Test that segments cover entire data range."""
        detector = BottomUpDetector(max_segments=5)
        result = detector.fit_detect(self.x, self.y)

        # First segment starts at 0
        assert result.segments[0].start == 0
        # Last segment ends at data length
        assert result.segments[-1].stop == len(self.y)

    def test_segment_statistics(self):
        """Test that segment statistics are computed."""
        detector = BottomUpDetector(max_segments=3)
        result = detector.fit_detect(self.x, self.y)

        for segment in result.segments:
            assert hasattr(segment, "slope")
            assert hasattr(segment, "offset")
            assert hasattr(segment, "std")


class TestPELTDetector:
    """Tests for PELTDetector (requires ruptures)."""

    @pytest.fixture(autouse=True)
    def check_ruptures(self):
        """Skip tests if ruptures is not installed."""
        pytest.importorskip("ruptures")

    def setup_method(self):
        """Set up test data."""
        self.x = np.arange(200, dtype=np.float64)
        self.y = np.concatenate(
            [
                np.linspace(0, 50, 100),
                np.linspace(50, 20, 100),
            ]
        )

    def test_basic_detection(self):
        """Test basic segment detection."""
        from trend_classifier.detectors import PELTDetector

        detector = PELTDetector(penalty=10)
        result = detector.fit_detect(self.x, self.y)

        assert isinstance(result, DetectionResult)
        assert len(result.segments) >= 1
        assert result.metadata["algorithm"] == "pelt"

    def test_different_models(self):
        """Test different cost models."""
        from trend_classifier.detectors import PELTDetector

        for model in ["l2", "l1"]:
            detector = PELTDetector(model=model, penalty=10)
            result = detector.fit_detect(self.x, self.y)
            assert len(result.segments) >= 1

    def test_auto_penalty(self):
        """Test automatic penalty calculation."""
        from trend_classifier.detectors import PELTDetector

        detector = PELTDetector(penalty=None)  # Auto penalty
        result = detector.fit_detect(self.x, self.y)

        assert len(result.segments) >= 1
        assert "penalty" in result.metadata


class TestDetectorRegistry:
    """Tests for detector registry functions."""

    def test_list_detectors(self):
        """Test listing available detectors."""
        detectors = list_detectors()
        assert "sliding_window" in detectors
        assert "bottom_up" in detectors

    def test_get_detector_sliding_window(self):
        """Test getting sliding window detector."""
        detector = get_detector("sliding_window", n=40)
        assert isinstance(detector, SlidingWindowDetector)
        assert detector.n == 40

    def test_get_detector_bottom_up(self):
        """Test getting bottom-up detector."""
        detector = get_detector("bottom_up", max_segments=10)
        assert isinstance(detector, BottomUpDetector)
        assert detector.max_segments == 10

    def test_get_detector_unknown(self):
        """Test error for unknown detector."""
        with pytest.raises(ValueError, match="Unknown detector"):
            get_detector("unknown_detector")

    def test_get_detector_pelt(self):
        """Test getting PELT detector."""
        try:
            import ruptures  # noqa: F401
        except ImportError:
            pytest.skip("PELT not available (ruptures not installed)")

        detector = get_detector("pelt", penalty=5)
        assert detector.name == "pelt"


class TestBaseDetector:
    """Tests for BaseDetector interface."""

    def test_is_abstract(self):
        """Test that BaseDetector cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_concrete_implementation(self):
        """Test that concrete implementations work."""
        detector = SlidingWindowDetector()
        assert isinstance(detector, BaseDetector)
        assert detector.name == "sliding_window"
