"""Loads Hugging Face transformers DINOv3 ViT checkpoints into Meta's DinoVisionTransformer."""

import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from dinov3.models.vision_transformer import DinoVisionTransformer

# Inverse of the renames in transformers' convert_dinov3_vit_to_hf.py, applied in order.
_RENAMES = [
    (r"^embeddings\.cls_token$", "cls_token"),
    (r"^embeddings\.register_tokens$", "storage_tokens"),
    (r"^embeddings\.mask_token$", "mask_token"),
    (r"^embeddings\.patch_embeddings\.", "patch_embed.proj."),
    (r"^layer\.(\d+)\.", r"blocks.\1."),
    (r"\.attention\.o_proj\.", ".attn.proj."),
    (r"\.layer_scale(\d)\.lambda1$", r".ls\1.gamma"),
    (r"\.mlp\.up_proj\.", ".mlp.fc1."),
    (r"\.mlp\.down_proj\.", ".mlp.fc2."),
]


def _rename(key: str) -> str:
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def load_hf_checkpoint(model: DinoVisionTransformer, path: str | Path) -> DinoVisionTransformer:
    """Loads `model.safetensors` from a Hugging Face DINOv3 ViT repo (or the repo directory) into `model`.

    Hugging Face splits the attention input projection into q, k and v and may drop the key bias; Meta's model fuses
    them into one qkv layer whose bias_mask zeroes the biases Hugging Face leaves out. RoPE periods are not stored on
    the Hub, so the model keeps the ones it computed from its config.
    """
    path = Path(path)
    hf_state = load_file(path / "model.safetensors" if path.is_dir() else path)
    expected = model.state_dict()
    state = {_rename(key): value for key, value in hf_state.items()}
    state["mask_token"] = state["mask_token"].reshape(expected["mask_token"].shape)
    state["rope_embed.periods"] = expected["rope_embed.periods"]

    for i in range(len(model.blocks)):
        attn = f"blocks.{i}.attention."
        projections = [f"{attn}{name}_proj" for name in "qkv"]
        weights = [state.pop(f"{proj}.weight") for proj in projections]
        biases = [state.pop(f"{proj}.bias", None) for proj in projections]
        qkv = f"blocks.{i}.attn.qkv."
        state[qkv + "weight"] = torch.cat(weights)
        if qkv + "bias" in expected:
            state[qkv + "bias"] = torch.cat(
                [torch.zeros_like(w[:, 0]) if b is None else b for w, b in zip(weights, biases)]
            )
        if qkv + "bias_mask" in expected:
            state[qkv + "bias_mask"] = torch.cat(
                [torch.full_like(w[:, 0], b is not None) for w, b in zip(weights, biases)]
            )

    model.load_state_dict(state, strict=True)
    return model
