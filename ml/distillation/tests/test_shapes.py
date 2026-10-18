"""TinyEEGNet architecture, fixed-shape, and starter-interface contracts."""

from __future__ import annotations

import inspect

import torch


def test_tinyeegnet_has_fixed_input_output_and_parameter_budget() -> None:
    from src.models.tinyeegnet import TinyEEGNet

    model = TinyEEGNet().eval()
    x = torch.randn(1, 1, 22, 500)
    with torch.inference_mode():
        logits = model(x)

    assert logits.shape == (1, 2)
    assert logits.dtype == x.dtype
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert 0 < trainable < 3000, f"TinyEEGNet has {trainable} trainable parameters"


def test_tinyeegnet_matches_the_agreed_deployable_architecture() -> None:
    from src.models.tinyeegnet import TinyEEGNet

    model = TinyEEGNet()
    convs = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    linears = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    pools = [layer for layer in model.modules() if isinstance(layer, torch.nn.AvgPool2d)]

    conv_specs = {(c.in_channels, c.out_channels, c.kernel_size, c.groups) for c in convs}
    assert (1, 8, (1, 64), 1) in conv_specs
    assert (8, 16, (22, 1), 8) in conv_specs
    assert (16, 16, (1, 16), 16) in conv_specs
    assert (16, 16, (1, 1), 1) in conv_specs
    assert len(linears) == 1
    assert (linears[0].in_features, linears[0].out_features) == (240, 2)
    assert [pool.kernel_size for pool in pools] == [(1, 4), (1, 8)]
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in model.modules()) == 2
    assert not any(isinstance(layer, (torch.nn.ELU, torch.nn.Softmax)) for layer in model.modules())


def test_starter_entry_point_signatures_and_outputs() -> None:
    from src.models.tinyeegnet import get_calibration_inputs, get_model

    model_signature = inspect.signature(get_model)
    calibration_signature = inspect.signature(get_calibration_inputs)

    assert list(model_signature.parameters) == ["input_shape"]
    assert model_signature.parameters["input_shape"].default == (1, 1, 22, 500)
    assert list(calibration_signature.parameters) == ["input_shape", "calibration_samples"]
    assert calibration_signature.parameters["input_shape"].default == (1, 1, 22, 500)
    assert calibration_signature.parameters["calibration_samples"].default == 200

    model = get_model()
    assert isinstance(model, torch.nn.Module)
    assert not model.training

    samples = get_calibration_inputs(calibration_samples=3)
    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(isinstance(sample, torch.Tensor) for sample in samples)
    assert all(sample.shape == (1, 1, 22, 500) for sample in samples)
    assert all(torch.isfinite(sample).all() for sample in samples)

