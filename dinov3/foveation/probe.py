"""Linear segmentation probe on DIS5K. It compares frozen features from dense, foveated and ConvNeXt DINOv3 backbones
at several input resolutions, scores them against the same masks, and times each backbone.

Each image is decoded once and resized for every variant. A linear layer per variant maps each feature-grid cell to
a foreground logit. It trains on a random sample of cells from each training image, against the mask's foreground
fraction in that cell. On the validation images its logits are upsampled bilinearly to a common scoring size (long
side --score-size) and scored by IoU (thresholded at 0.5) and MAE, averaged over images.

Speed is the median forward time on a 3:2 image at each variant's input size, in bf16, with compiled pooling for
the foveated variants. Adapters are merged into the weights, so adapted variants cost the same as foveated ones.

Variants, built from the Phase 1 run in --run, at each of --sizes unless noted:
    dense-{size}        the unmodified ViT, at each of --dense-sizes
    foveated-{size}     the run's foveation config, with start cells scaled with the image
    adapted-{size}      the same with the run's LoRA adapters
    random-{size}       the same with random split scores
    uniform{px}-{size}  no splits: every key pooled over start cells px wide, at the foveated start level and one finer
    convnext-{name}-{size}  DINOv3's distilled ConvNeXts (stride 32), for each of --convnexts

Example:
    python -m dinov3.foveation.probe --run data/runs/pilot_vits16 \\
        --train "data/dis5k/data/DIS_TR-*.parquet" --val "data/dis5k/data/DIS_VD-*.parquet"
"""

import argparse
import copy
import json
import logging
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import set_peft_model_state_dict
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveatedAttention, FoveationConfig, foveate
from dinov3.foveation.benchmark import measure
from dinov3.foveation.distill import foveation_config, make_student
from dinov3.foveation.phase0 import load_model, patch_aligned_size

logger = logging.getLogger("dinov3")

ALIGN = 32  # the ConvNeXt stride, and a multiple of the ViT patch size


class Segmentation(Dataset):
    """Original uint8 images [3, H, W] and masks [H, W] from DIS5K parquet files."""

    def __init__(self, files: str, limit: int | None = None):
        rows = load_dataset("parquet", data_files=files, split="train")
        self.rows = rows.select(range(min(limit, len(rows)))) if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        row = self.rows[i]
        image = v2.functional.pil_to_tensor(row["image"].convert("RGB"))
        return image, v2.functional.pil_to_tensor(row["label"].convert("L"))[0]


def resize(image: Tensor, long_side: int, interpolation=v2.InterpolationMode.BILINEAR) -> Tensor:
    size = patch_aligned_size(*image.shape[-2:], long_side, ALIGN)
    return v2.functional.resize(image, size, interpolation=interpolation, antialias=True)


@dataclass
class Variant:
    name: str
    size: int
    model: nn.Module
    grid: Callable[[Tensor], Tensor]
    """Feature grid [h, w, C] for a normalised image [1, 3, H, W]."""

    @torch.no_grad()
    def features(self, image: Tensor) -> Tensor:
        """Feature grid for an original uint8 image on the GPU."""
        x = make_base_transform()(resize(image, self.size, v2.InterpolationMode.BICUBIC))[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.grid(x).float()


def vit(name: str, size: int, model) -> Variant:
    def grid(x: Tensor) -> Tensor:
        tokens = model.forward_features(x)["x_norm_patchtokens"][0]
        return tokens.unflatten(0, (x.shape[-2] // model.patch_size, x.shape[-1] // model.patch_size))

    return Variant(name, size, model, grid)


def convnext(name: str, size: int, model) -> Variant:
    def grid(x: Tensor) -> Tensor:
        tokens = model(pixel_values=x).last_hidden_state[0, 1:]  # after the pooled token
        return tokens.unflatten(0, (x.shape[-2] // ALIGN, x.shape[-1] // ALIGN))

    return Variant(name, size, model, grid)


def build_variants(teacher, run: dict, adapter: dict, args) -> list[Variant]:
    def foveated(config: FoveationConfig):
        return foveate(copy.deepcopy(teacher), config)

    def adapted(config: FoveationConfig):
        student = make_student(teacher, config, run["rank"])
        set_peft_model_state_dict(student, adapter)
        return student.merge_and_unload()

    variants = [vit(f"dense-{size}", size, teacher) for size in args.dense_sizes]
    for size in args.sizes:
        # Keeps the start cells the same share of the image as in training
        level = run["start_level"] + round(math.log2(size / run["size"]))
        config = foveation_config(run | dict(start_level=level))
        variants += [
            vit(f"foveated-{size}", size, foveated(config)),
            vit(f"adapted-{size}", size, adapted(config)),
            vit(f"random-{size}", size, foveated(replace(config, score="random"))),
        ]
        for uniform in (level, level - 1):
            px = teacher.patch_size * 2**uniform
            uniform_config = replace(config, order="uniform", start_level=uniform, score_level=uniform)
            variants.append(vit(f"uniform{px}-{size}", size, foveated(uniform_config)))
    for name in args.convnexts:
        weights = Path(run["weights"]).parent / f"dinov3-convnext-{name}-pretrain-lvd1689m"
        model = transformers.AutoModel.from_pretrained(weights).cuda().eval().requires_grad_(False)
        variants += [convnext(f"convnext-{name}-{size}", size, model) for size in args.sizes]
    return variants


@torch.no_grad()
def sample_cells(variants: list[Variant], data: Segmentation, n_cells: int, folder: Path) -> None:
    """Saves the features and foreground fractions of `n_cells` random grid cells per image for each variant, in
    chunks of 100 images."""
    samples = {v.name: [] for v in variants}
    for i in range(len(data)):
        image, mask = (t.cuda() for t in data[i])
        for v in variants:
            grid = v.features(image)
            fraction = v2.functional.resize(mask[None].float() / 255, list(grid.shape[:2]), antialias=True)[0]
            pick = torch.randperm(fraction.numel(), device="cuda")[:n_cells]
            samples[v.name].append((grid.flatten(0, 1)[pick].half().cpu(), fraction.flatten()[pick].half().cpu()))
        if (i + 1) % 100 == 0 or i + 1 == len(data):
            for name, chunk in samples.items():
                torch.save([torch.cat(t) for t in zip(*chunk)], folder / f"{name}.{i}.pt")
                chunk.clear()
            logger.info(f"Sampled cells from {i + 1}/{len(data)} training images")


def fit_probe(folder: Path, name: str, epochs: int, lr: float, batch: int = 8192) -> nn.Linear:
    chunks = [torch.load(path) for path in sorted(folder.glob(f"{name}.*.pt"))]
    features, fraction = (torch.cat(t).cuda().float() for t in zip(*chunks))
    # Standardising the features lets one learning rate suit every backbone
    mean, std = features.mean(0), features.std(0) + 1e-6
    probe = nn.Linear(features.shape[1], 1).cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    steps = epochs * math.ceil(len(features) / batch)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(features), device="cuda").split(batch):
            logits = probe((features[idx] - mean) / std)[:, 0]
            F.binary_cross_entropy_with_logits(logits, fraction[idx]).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
    with torch.no_grad():  # folds the standardisation into the layer
        probe.bias -= (probe.weight * mean / std).sum()
        probe.weight /= std
    return probe.requires_grad_(False)


@torch.no_grad()
def score(variants: list[Variant], probes: dict[str, nn.Linear], data: Segmentation, size: int) -> dict[str, dict]:
    ious, maes = ({v.name: [] for v in variants} for _ in range(2))
    for i in range(len(data)):
        image, mask = (t.cuda() for t in data[i])
        truth = resize(mask[None].float(), size)[0] > 127
        for v in variants:
            logits = probes[v.name](v.features(image))[..., 0]
            foreground = F.interpolate(logits[None, None], size=truth.shape, mode="bilinear")[0, 0].sigmoid()
            predicted = foreground > 0.5
            ious[v.name].append(((predicted & truth).sum() / (predicted | truth).sum().clamp(min=1)).item())
            maes[v.name].append((foreground - truth.float()).abs().mean().item())
    return {name: dict(iou=sum(ious[name]) / len(data), mae=sum(maes[name]) / len(data)) for name in ious}


def speed(variant: Variant, min_run_time: float) -> dict[str, float]:
    for module in variant.model.modules():
        if isinstance(module, FoveatedAttention):
            module.config = replace(module.config, compile=True)
    x = torch.randn(1, 3, *patch_aligned_size(2, 3, variant.size, ALIGN), device="cuda")
    timing = measure(variant.grid, x, torch.bfloat16, min_run_time)
    return dict(time_ms=timing["time_ms"], peak_activation_mb=timing["peak_activation_mb"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Output directory of a distill.py run")
    parser.add_argument("--train", required=True, help="Glob of DIS5K parquet files to fit the probes on")
    parser.add_argument("--val", required=True, help="Glob of DIS5K parquet files to score the probes on")
    parser.add_argument("--limit", type=int, help="Use at most this many training and validation images")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1920, 3840])
    parser.add_argument("--dense-sizes", type=int, nargs="+", default=[960, 1280, 1600, 1920, 2560, 3840])
    parser.add_argument("--convnexts", nargs="*", default=["tiny", "small"])
    parser.add_argument("--score-size", type=int, default=1920, help="Long side of the masks the probes are scored on")
    parser.add_argument("--cells", type=int, default=256, help="Grid cells sampled per training image")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-run-time", type=float, default=2.0, help="Seconds of timing per variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Default: probe_dis5k.csv in the run directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    run = json.loads((args.run / "config.json").read_text())
    adapter = torch.load(args.run / "adapter.pt")
    teacher = load_model(run["arch"], run["weights"]).cuda().eval().requires_grad_(False)
    variants = build_variants(teacher, run, adapter, args)
    logger.info(f"Variants: {[v.name for v in variants]}")

    with tempfile.TemporaryDirectory() as folder:
        sample_cells(variants, Segmentation(args.train, args.limit), args.cells, Path(folder))
        probes = {v.name: fit_probe(Path(folder), v.name, args.epochs, args.lr) for v in variants}
    scores = score(variants, probes, Segmentation(args.val, args.limit), args.score_size)

    rows = [dict(variant=v.name, size=v.size) | scores[v.name] | speed(v, args.min_run_time) for v in variants]
    table = pd.DataFrame(rows)
    table.to_csv(args.out or args.run / "probe_dis5k.csv", index=False)
    print(table.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
