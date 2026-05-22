import numpy as np
import pytest
import torch
import torch.nn as nn
from moove.utils.segment_utils import segment_ml
from moove.models.ConvMLP import ConvMLP


class AlwaysSyllableModel(nn.Module):
    """Mock model that always predicts syllable (logit > 0)."""
    def forward(self, x):
        return torch.tensor([[5.0]])


class AlwaysSilenceModel(nn.Module):
    """Mock model that always predicts silence (logit < 0)."""
    def forward(self, x):
        return torch.tensor([[-5.0]])


class AlternatingSyllableModel(nn.Module):
    """Mock model that alternates between syllable and silence."""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        # Produce blocks: 20 chunks syllable, 20 chunks silence
        block = (self.call_count // 20) % 2
        return torch.tensor([[5.0]]) if block == 0 else torch.tensor([[-5.0]])


class TestSegmentML:
    """Tests for the sliding-window segmentation pipeline."""

    @pytest.fixture
    def default_metadata(self):
        return {'mean': 0.0, 'std': 1.0}

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    def test_pure_silence_produces_no_segments(self, default_metadata, device):
        model = AlwaysSilenceModel()
        raw = np.zeros(44100, dtype=np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
        )
        assert len(onsets) == 0
        assert len(offsets) == 0

    def test_onsets_before_offsets(self, default_metadata, device):
        model = AlternatingSyllableModel()
        raw = np.random.randn(44100).astype(np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
        )
        for on, off in zip(onsets, offsets):
            assert on < off, f"Onset {on} should be before offset {off}"

    def test_onset_offset_count_matches(self, default_metadata, device):
        model = AlternatingSyllableModel()
        raw = np.random.randn(44100 * 2).astype(np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
        )
        assert len(onsets) == len(offsets)

    def test_segments_have_positive_duration(self, default_metadata, device):
        model = AlternatingSyllableModel()
        raw = np.random.randn(44100 * 2).astype(np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
            min_syllable_length=0.001,
        )
        for on, off in zip(onsets, offsets):
            assert off - on > 0

    def test_real_model_runs(self, default_metadata, device):
        """Smoke test: segment_ml should run with the actual ConvMLP model."""
        model = ConvMLP(input_size=192)
        model.eval()
        raw = np.random.randn(44100).astype(np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
        )
        assert len(onsets) == len(offsets)

    def test_short_audio_does_not_crash(self, default_metadata, device):
        """Very short audio (less than hist_size chunks) should not crash."""
        model = AlwaysSyllableModel()
        raw = np.zeros(100, dtype=np.float32)
        onsets, offsets = segment_ml(
            model, default_metadata, device, raw,
            sampling_rate=44100, chunk_size=64, hist_size=3,
        )
        assert len(onsets) == len(offsets)
