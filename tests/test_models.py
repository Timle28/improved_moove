import pytest
import torch
from moove.models.ConvMLP import ConvMLP
from moove.models.CNN import CNN


class TestConvMLP:
    """Tests for the binary segmentation model."""

    def test_output_shape_default(self):
        model = ConvMLP(input_size=192)
        model.eval()
        x = torch.randn(4, 192)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_single_sample(self):
        model = ConvMLP(input_size=192)
        model.eval()
        x = torch.randn(1, 192)
        out = model(x)
        assert out.shape == (1, 1)

    def test_different_input_sizes(self):
        for size in [64, 128, 192, 256, 512]:
            model = ConvMLP(input_size=size)
            model.eval()
            x = torch.randn(2, size)
            out = model(x)
            assert out.shape == (2, 1), f"Failed for input_size={size}"

    def test_output_is_logit(self):
        """Last layer should be Linear (raw logits), not Sigmoid."""
        model = ConvMLP(input_size=192)
        last_layer = list(model.mlp_layers.children())[-1]
        assert isinstance(last_layer, torch.nn.Linear), \
            f"Expected Linear as last layer, got {type(last_layer).__name__}"

    def test_gradient_flow(self):
        model = ConvMLP(input_size=192)
        x = torch.randn(4, 192)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCNN:
    """Tests for the syllable classification model."""

    def test_output_shape(self):
        input_shape = (1, 32, 16)
        model = CNN(input_shape=input_shape, num_classes=5)
        model.eval()
        x = torch.randn(4, *input_shape)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_sample(self):
        input_shape = (1, 32, 16)
        model = CNN(input_shape=input_shape, num_classes=3)
        model.eval()
        x = torch.randn(1, *input_shape)
        out = model(x)
        assert out.shape == (1, 3)

    def test_different_class_counts(self):
        input_shape = (1, 32, 16)
        for n_classes in [2, 5, 10, 26]:
            model = CNN(input_shape=input_shape, num_classes=n_classes)
            model.eval()
            x = torch.randn(2, *input_shape)
            out = model(x)
            assert out.shape == (2, n_classes)

    def test_different_spectrogram_sizes(self):
        """Spectrograms can vary in size depending on FFT parameters."""
        for h, w in [(16, 8), (32, 16), (64, 32), (128, 21)]:
            input_shape = (1, h, w)
            model = CNN(input_shape=input_shape, num_classes=4)
            model.eval()
            x = torch.randn(2, *input_shape)
            out = model(x)
            assert out.shape == (2, 4), f"Failed for spectrogram size {h}x{w}"

    def test_gradient_flow(self):
        input_shape = (1, 32, 16)
        model = CNN(input_shape=input_shape, num_classes=5)
        x = torch.randn(4, *input_shape)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
