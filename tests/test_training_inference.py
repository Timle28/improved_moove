"""Tests for PyTorch model training and inference pipelines.

Tests marked ``needs_pretrained`` are skipped when
``tests/fixtures/pretrained/`` is absent or empty.
"""
import os

import numpy as np
import pytest
import torch

from moove.models.CNN import CNN
from moove.models.ConvMLP import ConvMLP


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pretrained")

needs_pretrained = pytest.mark.skipif(
    not os.path.isdir(FIXTURES_DIR) or not os.listdir(FIXTURES_DIR) if os.path.isdir(FIXTURES_DIR) else True,
    reason="Pretrained model fixtures not yet provided (tests/fixtures/pretrained/)",
)


# =====================================================================
# Unit tests – model forward pass
# =====================================================================

class TestSegmentationModelForward:
    """ConvMLP is used for segmentation (binary output)."""

    def test_forward_pass_runs(self):
        model = ConvMLP(input_size=64)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape[0] == 4

    def test_output_in_logit_range(self):
        model = ConvMLP(input_size=64)
        x = torch.randn(8, 64)
        out = model(x)
        # Before sigmoid, logits can be any real number – just check no NaN
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        model = ConvMLP(input_size=64)
        x = torch.randn(4, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


class TestClassificationModelForward:
    """CNN is used for classification (multi-class output)."""

    def test_forward_pass_runs(self):
        model = CNN(input_shape=(1, 33, 22), num_classes=5)
        x = torch.randn(4, 1, 33, 22)
        out = model(x)
        assert out.shape == (4, 5)

    def test_softmax_sums_to_one(self):
        model = CNN(input_shape=(1, 33, 22), num_classes=5)
        x = torch.randn(4, 1, 33, 22)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_gradient_flows(self):
        model = CNN(input_shape=(1, 33, 22), num_classes=5)
        x = torch.randn(4, 1, 33, 22)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# =====================================================================
# Training loop smoke tests
# =====================================================================

class TestSegmentationTrainingSmoke:
    """Verify that a minimal training loop converges (loss decreases)."""

    def test_one_epoch_loss_decreases(self):
        torch.manual_seed(42)
        model = ConvMLP(input_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Synthetic: 32 samples, binary labels
        X = torch.randn(32, 64)
        y = torch.randint(0, 2, (32, 1)).float()

        model.train()
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over 5 steps on same data
        assert losses[-1] < losses[0]


class TestClassificationTrainingSmoke:
    """Verify that a minimal classification training loop works."""

    def test_one_epoch_loss_decreases(self):
        torch.manual_seed(42)
        num_classes = 4
        model = CNN(input_shape=(1, 33, 22), num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        X = torch.randn(32, 1, 33, 22)
        y = torch.randint(0, num_classes, (32,))

        model.train()
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]


# =====================================================================
# Checkpoint save / load round-trip
# =====================================================================

class TestCheckpointRoundTrip:
    def test_save_load_segmentation(self, tmp_path):
        model = ConvMLP(input_size=64)
        metadata = {"chunk_size": 64, "hist_size": 3}

        path = tmp_path / "seg_model.pth"
        torch.save({"model": model, "metadata": metadata}, str(path))

        loaded = torch.load(str(path), weights_only=False)
        assert "model" in loaded
        assert "metadata" in loaded
        assert loaded["metadata"]["chunk_size"] == 64

        # Loaded model should produce same output
        x = torch.randn(1, 64)
        model.eval()
        loaded["model"].eval()
        with torch.no_grad():
            orig_out = model(x)
            loaded_out = loaded["model"](x)
        assert torch.allclose(orig_out, loaded_out)

    def test_save_load_classification(self, tmp_path):
        model = CNN(input_shape=(1, 33, 22), num_classes=5)
        metadata = {"int_to_label": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}}

        path = tmp_path / "class_model.pth"
        torch.save({"model": model, "metadata": metadata}, str(path))

        loaded = torch.load(str(path), weights_only=False)
        assert loaded["metadata"]["int_to_label"][0] == "a"


# =====================================================================
# Pretrained model tests
# =====================================================================

@needs_pretrained
class TestPretrainedSegmentation:
    """Tests that use a real pretrained segmentation model."""

    def test_load_pretrained_seg_model(self):
        path = os.path.join(FIXTURES_DIR, "test_seg_model.pth")
        checkpoint = torch.load(path, weights_only=False)
        model = checkpoint["model"]
        model.eval()
        x = torch.randn(1, 1, checkpoint["metadata"]["chunk_size"])
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1

    def test_seg_inference_on_synthetic_audio(self):
        path = os.path.join(FIXTURES_DIR, "test_seg_model.pth")
        checkpoint = torch.load(path, weights_only=False)
        model = checkpoint["model"]
        model.eval()

        chunk_size = checkpoint["metadata"]["chunk_size"]
        hist_size = checkpoint["metadata"]["hist_size"]

        # Simulate hist_size chunks of audio
        audio = torch.randn(1, 1, chunk_size * hist_size)
        with torch.no_grad():
            out = torch.sigmoid(model(audio))
        assert 0.0 <= out.item() <= 1.0


@needs_pretrained
class TestPretrainedClassification:
    """Tests that use a real pretrained classification model."""

    def test_load_pretrained_class_model(self):
        path = os.path.join(FIXTURES_DIR, "test_class_model.pth")
        checkpoint = torch.load(path, weights_only=False)
        model = checkpoint["model"]
        model.eval()
        x = torch.randn(1, 1, 33, 22)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] > 1  # multi-class

    def test_class_inference_produces_valid_label(self):
        path = os.path.join(FIXTURES_DIR, "test_class_model.pth")
        checkpoint = torch.load(path, weights_only=False)
        model = checkpoint["model"]
        model.eval()
        meta = checkpoint["metadata"]
        int_to_label = meta["int_to_label"]

        x = torch.randn(1, 1, 33, 22)
        with torch.no_grad():
            out = model(x)
        predicted = int_to_label[torch.argmax(out).item()]
        assert isinstance(predicted, str)
        assert len(predicted) == 1  # single character label
