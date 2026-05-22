import numpy as np
import pytest
from moove.utils.label_utils import normalize_spectrogram


class TestNormalizeSpectrogram:
    def test_zero_mean(self):
        spec = np.random.rand(64, 32)
        result = normalize_spectrogram(spec)
        assert abs(result.mean()) < 1e-6

    def test_unit_variance(self):
        spec = np.random.rand(64, 32)
        result = normalize_spectrogram(spec)
        assert abs(result.std() - 1.0) < 1e-6

    def test_constant_input(self):
        """A constant spectrogram (std=0) should be returned as-is."""
        spec = np.ones((32, 16)) * 5.0
        result = normalize_spectrogram(spec)
        np.testing.assert_array_equal(result, spec)

    def test_shape_preserved(self):
        spec = np.random.rand(128, 64)
        result = normalize_spectrogram(spec)
        assert result.shape == spec.shape
