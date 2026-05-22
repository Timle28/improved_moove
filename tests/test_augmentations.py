import numpy as np
import pytest
from moove.utils.training_utils import (
    augment_spectrogram,
    add_noise_to_spectrogram,
    frequency_mask,
    time_mask,
    dynamic_range_compression,
    DEFAULT_AUGMENTATION_PARAMS,
)


@pytest.fixture
def sample_spectrogram():
    np.random.seed(0)
    return np.random.rand(64, 32).astype(np.float32)


class TestAddNoise:
    def test_shape_preserved(self, sample_spectrogram):
        result = add_noise_to_spectrogram(sample_spectrogram)
        assert result.shape == sample_spectrogram.shape

    def test_noise_is_small(self, sample_spectrogram):
        result = add_noise_to_spectrogram(sample_spectrogram, noise_level=0.0001)
        diff = np.abs(result - sample_spectrogram)
        assert diff.max() < 0.01


class TestFrequencyMask:
    def test_shape_preserved(self, sample_spectrogram):
        result = frequency_mask(sample_spectrogram)
        assert result.shape == sample_spectrogram.shape

    def test_does_not_modify_input(self, sample_spectrogram):
        original = sample_spectrogram.copy()
        frequency_mask(sample_spectrogram)
        np.testing.assert_array_equal(sample_spectrogram, original)

    def test_some_values_changed(self, sample_spectrogram):
        np.random.seed(1)
        result = frequency_mask(sample_spectrogram, F=20)
        assert not np.array_equal(result, sample_spectrogram)


class TestTimeMask:
    def test_shape_preserved(self, sample_spectrogram):
        result = time_mask(sample_spectrogram)
        assert result.shape == sample_spectrogram.shape

    def test_does_not_modify_input(self, sample_spectrogram):
        original = sample_spectrogram.copy()
        time_mask(sample_spectrogram)
        np.testing.assert_array_equal(sample_spectrogram, original)

    def test_some_values_changed(self, sample_spectrogram):
        np.random.seed(1)
        result = time_mask(sample_spectrogram, T=20)
        assert not np.array_equal(result, sample_spectrogram)


class TestDynamicRangeCompression:
    def test_shape_preserved(self, sample_spectrogram):
        result = dynamic_range_compression(sample_spectrogram)
        assert result.shape == sample_spectrogram.shape

    def test_no_nans(self, sample_spectrogram):
        result = dynamic_range_compression(sample_spectrogram)
        assert not np.any(np.isnan(result))


class TestAugmentSpectrogram:
    def test_shape_preserved(self, sample_spectrogram):
        result = augment_spectrogram(sample_spectrogram)
        assert result.shape == sample_spectrogram.shape

    def test_sometimes_unchanged(self):
        """With 80% probability the spectrogram should be returned unchanged."""
        spec = np.ones((32, 16), dtype=np.float32)
        np.random.seed(0)
        unchanged_count = sum(
            np.array_equal(augment_spectrogram(spec.copy()), spec)
            for _ in range(100)
        )
        assert unchanged_count > 50, "Augmentation should leave most samples unchanged"

    def test_disabled_returns_unchanged(self, sample_spectrogram):
        """When enabled=False, spectrogram is always returned unchanged."""
        params = dict(DEFAULT_AUGMENTATION_PARAMS, enabled=False)
        original = sample_spectrogram.copy()
        for _ in range(20):
            result = augment_spectrogram(original.copy(), aug_params=params)
            np.testing.assert_array_equal(result, original)

    def test_custom_probability(self, sample_spectrogram):
        """probability=1.0 should always augment."""
        params = dict(DEFAULT_AUGMENTATION_PARAMS, probability=1.0)
        changed = 0
        np.random.seed(42)
        for _ in range(20):
            result = augment_spectrogram(sample_spectrogram.copy(), aug_params=params)
            if not np.array_equal(result, sample_spectrogram):
                changed += 1
        assert changed == 20

    def test_custom_params_passed_through(self):
        """Custom noise_level should produce correspondingly larger perturbations."""
        spec = np.ones((32, 16), dtype=np.float32)
        params = dict(DEFAULT_AUGMENTATION_PARAMS, probability=1.0, noise_level=10.0)
        np.random.seed(0)
        import random
        random.seed(0)
        # Force noise augmentation by seeding; run enough times to hit it
        diffs = []
        for _ in range(50):
            result = augment_spectrogram(spec.copy(), aug_params=params)
            diffs.append(np.abs(result - spec).max())
        assert max(diffs) > 1.0, "High noise_level should produce large perturbations"
