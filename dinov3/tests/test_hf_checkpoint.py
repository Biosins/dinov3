from pathlib import Path

import pytest
import torch

from dinov3.hub.backbones import dinov3_vitl16
from dinov3.hub.hf_checkpoint import load_hf_checkpoint

WEIGHTS = Path(__file__).parents[2] / "weights" / "dinov3-vitl16-pretrain-lvd1689m"


@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"download the Hugging Face weights to {WEIGHTS}")
def test_converted_vitl16_matches_transformers():
    transformers = pytest.importorskip("transformers")
    hf_model = transformers.AutoModel.from_pretrained(WEIGHTS).eval()
    model = load_hf_checkpoint(dinov3_vitl16(pretrained=False), WEIGHTS).eval()

    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 320)
    with torch.no_grad():
        expected = hf_model(pixel_values=image).last_hidden_state
        actual = model.forward_features(image)
    torch.testing.assert_close(actual["x_norm_clstoken"], expected[:, 0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        actual["x_norm_patchtokens"], expected[:, 1 + model.n_storage_tokens :], atol=1e-4, rtol=1e-4
    )
