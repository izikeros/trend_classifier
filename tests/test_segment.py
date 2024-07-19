import pytest
from pydantic import ValidationError
from trend_classifier import Segment


class TestSegment:
    def test_segment__fails_validation_due_to_missing_input(self):
        with pytest.raises(ValidationError):
            Segment()

    def test_segment__fails_validation_due_to_missing_start(self):
        s = Segment(start=0, stop=0, slope=0.1, offset=0.1)
        assert isinstance(s, Segment)

    def test_segment__repr__runs(self):
        repr(Segment(start=0, stop=0, slope=0.1, offset=0.1))

    def test_segment__str__runs(self):
        str(Segment(start=0, stop=0, slope=0.1, offset=0.1))
