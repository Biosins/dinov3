from pathlib import Path

import pytest
import torch

from dinov3.hub import backbones
from dinov3.hub.hf_checkpoint import load_hf_checkpoint

WEIGHTS = Path(__file__).parents[2] / "weights"


@pytest.mark.parametrize("arch", ["vits16", "vitb16", "vitl16"])
def test_converted_checkpoint_matches_transformers(arch):
    path = WEIGHTS / f"dinov3-{arch}-pretrain-lvd1689m"
    if not path.exists():
        pytest.skip(f"download the Hugging Face weights to {path}")
    transformers = pytest.importorskip("transformers")
    hf_model = transformers.AutoModel.from_pretrained(path).eval()
    model = load_hf_checkpoint(getattr(backbones, f"dinov3_{arch}")(pretrained=False), path).eval()

    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 320)
    with torch.no_grad():
        expected = hf_model(pixel_values=image).last_hidden_state
        actual = model.forward_features(image)
    torch.testing.assert_close(actual["x_norm_clstoken"], expected[:, 0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        actual["x_norm_patchtokens"], expected[:, 1 + model.n_storage_tokens :], atol=1e-4, rtol=1e-4
    )
