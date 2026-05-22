import numpy as np
import pytest
from moove.utils.audio_utils import (
    seconds_to_index,
    seconds_to_chunk_index,
    index_to_seconds,
    decibel,
)


class TestSecondsToIndex:
    def test_basic_conversion(self):
        # 100 ms at 44100 Hz → 100 * 44100 / 1000 = 4410
        assert seconds_to_index(100, 44100) == 4410

    def test_zero(self):
        assert seconds_to_index(0, 44100) == 0

    def test_one_second_in_ms(self):
        # Input is in milliseconds: 1000 ms at 44100 Hz → 44100
        assert seconds_to_index(1000, 44100) == 44100


class TestSecondsToChunkIndex:
    def test_basic(self):
        # 0.03s at 44100 Hz, chunk_size=64 → (0.03 * 44100) / 64 = 20.671..
        result = seconds_to_chunk_index(0.03, 64, 44100)
        assert result == 20.0

    def test_zero(self):
        assert seconds_to_chunk_index(0, 64, 44100) == 0


class TestIndexToSeconds:
    def test_basic(self):
        # index=100, chunk_size=64, sr=44100 → (100*64)/44100 ≈ 0.1451
        result = index_to_seconds(100, 64, 44100)
        assert abs(result - (100 * 64 / 44100)) < 1e-10

    def test_zero(self):
        assert index_to_seconds(0, 64, 44100) == 0.0

    def test_roundtrip_with_chunk_index(self):
        """Converting seconds→chunk_index→seconds should be roughly consistent."""
        seconds = 0.5
        chunk_size, sr = 64, 44100
        chunk_idx = seconds_to_chunk_index(seconds, chunk_size, sr)
        recovered = index_to_seconds(chunk_idx, chunk_size, sr)
        assert abs(recovered - seconds) < (chunk_size / sr)


class TestDecibel:
    def test_reference_is_zero_db(self):
        x = np.array([1.0])
        assert decibel(x)[0] == pytest.approx(0.0)

    def test_doubles_amplitude(self):
        x = np.array([2.0])
        assert decibel(x)[0] == pytest.approx(20 * np.log10(2.0))

    def test_near_zero_clipping(self):
        """Values near zero should be clipped, not produce -inf."""
        x = np.array([0.0, 1e-20, 1e-10])
        result = decibel(x)
        assert np.all(np.isfinite(result))

    def test_shape_preserved(self):
        x = np.random.rand(100)
        assert decibel(x).shape == (100,)

    def test_does_not_modify_input(self):
        x = np.array([0.0, 1.0, 2.0])
        x_copy = x.copy()
        decibel(x)
        np.testing.assert_array_equal(x, x_copy)
