import sys

import pytest
import torch
import torch.nn.functional as F

from scripts.benchmark_nyu import MODEL_NAME, NYUInputs, ROOT, build_model, model_config, summary
from model.deform_conv_backend import modulated_deform_conv2d


def test_backend_zero_offset_matches_regular_convolution():
    torch.manual_seed(7)
    x = torch.randn(1, 2, 4, 5)
    weight = torch.randn(3, 2, 3, 3)
    offset = torch.zeros(1, 18, 4, 5)
    mask = torch.ones(1, 9, 4, 5)
    actual = modulated_deform_conv2d(x, offset, mask, weight, None, padding=1)
    torch.testing.assert_close(actual, F.conv2d(x, weight, padding=1), rtol=1e-5, atol=1e-5)


def test_fractional_noncontiguous_offset_matches_bilinear_sampling():
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    full = torch.zeros(1, 18, 4, 10)
    offset = full[:, 4:6, :, ::2]
    assert not offset.is_contiguous()
    offset[:, 0] = 0.25
    offset[:, 1] = -0.3
    actual = modulated_deform_conv2d(x, offset, torch.ones_like(x), torch.ones(1, 1, 1, 1), None)
    y, z = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    grid = torch.stack((2 * (z - 0.3) / 4 - 1, 2 * (y + 0.25) / 3 - 1), dim=-1).unsqueeze(0)
    expected = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_random_model_never_loads_pretrained_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random inference must not load or download weights")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    torch.set_num_threads(1)
    args = model_config(2023, ROOT / "data/nyudepthv2_h5")
    model = build_model(args)
    assert args.from_scratch and args.dcn_backend == "torchvision"
    assert args.prop_time == (6 if MODEL_NAME == "CompletionFormer" else 18)
    assert sum(p.numel() for p in model.parameters()) > 1_000_000
    assert "apex" not in sys.modules and "DCN" not in sys.modules
    assert "mmcv" not in sys.modules and "mmseg" not in sys.modules
    assert torch.count_nonzero(model.prop_layer.conv_offset_aff.weight) == 0
    assert torch.equal(model.prop_layer.w, torch.ones_like(model.prop_layer.w))


def test_local_nyu_input_is_deterministic_and_has_500_points():
    dataset = NYUInputs(ROOT / "data/nyudepthv2_h5", seed=2023)
    sample = dataset[0]
    repeat = dataset[0]
    assert len(dataset) == 654
    assert sample["rgb"].shape == (1, 3, 228, 304)
    assert sample["dep"].shape == (1, 1, 228, 304)
    assert (sample["dep"] > 0).sum() == 500
    assert torch.equal(sample["dep"], repeat["dep"])


def test_summary_and_backend_validation():
    result = summary([10, 20, 30])
    assert result["mean_ms"] == 20 and result["p95_ms"] == 29 and result["fps"] == 50
    with pytest.raises(ValueError, match="Unsupported DCN"):
        modulated_deform_conv2d(None, None, None, None, None, backend="unknown")
