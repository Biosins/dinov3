"""Phase 1: distil dense DINOv3 into a foveated student with LoRA adapters on the attention projections.

Teacher: the frozen dense model. Student: the same weights, foveated, plus LoRA adapters on every block's qkv and
proj (via peft); everything else stays frozen. Loss: 1 - cosine similarity between student and teacher final
normalised features, for the CLS token plus the mean over patch tokens. Gradients reach the adapters through the
pooled keys and values and the final attention, but not through the layout choice.

Each training image gets a long side drawn from --sizes and, with --ratios, a budget of K/N drawn log-uniformly
between the two ratios, from the smallest start cells that leave at least half of it for splits. Held-out images
cycle through fixed sizes and ratios, so their scores stay comparable.

Every --eval-every images, student and teacher are compared end to end on the held-out images, starting with the
untrained student at step 0. The best adapter so far is kept; training stops after --patience evaluations without
improvement. Adapters and metrics go to --out.

Example:
    python -m dinov3.foveation.distill --arch vits16 --weights weights/dinov3-vits16-pretrain-lvd1689m \\
        --train data/div2k/DIV2K_train_HR --val data/div2k/DIV2K_valid_HR --out data/runs/pilot
"""

import argparse
import copy
import json
import logging
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import datasets
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveationConfig, foveate, reconfigure, start_cells
from dinov3.foveation.phase0 import ARCHS, IMAGE_SUFFIXES, load_model, resize_image

logger = logging.getLogger("dinov3")


class Images(Dataset):
    """uint8 images from a folder or from parquet files with "image" and "image_name" columns (a glob, as DIS5K on
    Hugging Face), resized to a long side from `sizes`. With `augment`, the size is drawn at random and images are
    flipped at random; without, image i gets size i mod len(sizes), so held-out conditions stay fixed. Images named
    in `exclude` are left out."""

    def __init__(
        self,
        source: str,
        sizes: list[int],
        patch_size: int,
        augment: bool,
        limit: int | None = None,
        exclude: frozenset[str] = frozenset(),
    ):
        if Path(source).is_dir():
            self.rows = None
            paths = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            self.items = [p for p in paths if p.name not in exclude]
        else:  # rows stay memory-mapped; only the kept indices are held
            rows = load_dataset("parquet", data_files=source, split="train")
            self.rows = rows.cast_column("image", datasets.Image(decode=False))
            self.items = [i for i, name in enumerate(rows["image_name"]) if name not in exclude]
        self.items = self.items[:limit]
        self.sizes, self.patch_size, self.augment = sizes, patch_size, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tensor:
        item = self.items[i]
        size = random.choice(self.sizes) if self.augment else self.sizes[i % len(self.sizes)]
        if self.rows is not None:
            encoded = torch.frombuffer(bytearray(self.rows[item]["image"]["bytes"]), dtype=torch.uint8)
            item = decode_image(encoded, mode=ImageReadMode.RGB)
        image = resize_image(item, size, self.patch_size)
        return image.flip(-1) if self.augment and random.random() < 0.5 else image


def budget_config(base: FoveationConfig, H: int, W: int, ratio: float, n_prefix: int) -> FoveationConfig:
    """base with a budget of ratio x N keys (a multiple of 64) for an H x W patch grid, from the smallest start
    cells that leave at least half of it for splits."""
    budget = max(64, round(ratio * H * W / 64) * 64)
    level = base.start_level
    while start_cells(H, W, level) + n_prefix > budget // 2:
        level += 1
    return replace(base, budget=budget, start_level=level, score_level=level)


def held_out_ratio(i: int, ratios: tuple[float, float], steps: int = 4) -> float:
    """K/N for held-out image i: `steps` fixed values spaced evenly in log between the training extremes."""
    low, high = ratios
    return low * (high / low) ** ((i % steps) / (steps - 1))


def features(model, image: Tensor) -> tuple[Tensor, Tensor]:
    """Final normalised CLS token [B, C] and patch tokens [B, N, C] for a uint8 image batch on the GPU."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward_features(make_base_transform()(image))
    return out["x_norm_clstoken"].float(), out["x_norm_patchtokens"].float()


def distance(student: tuple[Tensor, Tensor], teacher: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """1 - cosine similarity for the CLS token and averaged over patch tokens."""
    return tuple(1 - F.cosine_similarity(s, t, dim=-1).mean() for s, t in zip(student, teacher))


def foveation_config(run: dict) -> FoveationConfig:
    """The foveation config of a run, from its command-line arguments or saved config.json."""
    return FoveationConfig(
        budget=run["budget"],
        start_level=run["start_level"],
        score_level=run["start_level"],
        order=run["order"],
        score=run["score"],
        scoring=run.get("scoring", "full"),  # runs before beam scoring did not record it
    )


def lora_config(rank: int) -> LoraConfig:
    return LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=r"blocks\.\d+\.attn\.(qkv|proj)")


def make_student(teacher, config: FoveationConfig, rank: int):
    """A foveated copy of the teacher with LoRA adapters on every block's qkv and proj."""
    return get_peft_model(foveate(copy.deepcopy(teacher), config), lora_config(rank))


def with_adapters(model, run: Path):
    """A copy of a dense model with a run's trained adapters merged into its weights, ready to foveate."""
    rank = json.loads((run / "config.json").read_text())["rank"]
    student = get_peft_model(copy.deepcopy(model), lora_config(rank))
    adapter = torch.load(run / "adapter.pt", map_location=next(model.parameters()).device)
    set_peft_model_state_dict(student, adapter)
    return student.merge_and_unload()


@torch.no_grad()
def evaluate(student, teacher, images: Images, configure: Callable[[int, Tensor], None] | None = None) -> dict:
    """Mean CLS and patch cosine of student to teacher; configure(i, image) can set the student up per image."""
    cls, patches = [], []
    for i, image in enumerate(images):
        image = image[None].cuda()
        if configure:
            configure(i, image)
        cls_distance, patch_distance = distance(features(student, image), features(teacher, image))
        cls.append(1 - cls_distance.item())
        patches.append(1 - patch_distance.item())
    return dict(cls_cosine=sum(cls) / len(cls), patch_cosine=sum(patches) / len(patches))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vits16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--train", required=True, help="Folder of training images, or a glob of parquet files")
    parser.add_argument("--val", required=True, help="Folder of held-out images, or a glob of parquet files")
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--exclude", type=Path, help="Folder whose image names are left out of both sets")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1920], help="Long sides in pixels, one per image")
    parser.add_argument("--budget", type=int, default=1024, help="Keys per head, unless --ratios is given")
    parser.add_argument(
        "--ratios", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Draw K/N per image, log-uniform in between"
    )
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--order", choices=["per_level", "global", "tree"], default="tree")
    parser.add_argument("--score", choices=["attention", "detail", "bound", "variance", "random"], default="variance")
    parser.add_argument("--scoring", choices=["full", "beam"], default="beam")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--images", type=int, default=2000, help="Most training images seen in total")
    parser.add_argument("--accumulate", type=int, default=4, help="Images per optimiser step")
    parser.add_argument("--warmup", type=int, default=50, help="Optimiser steps of linear warm-up")
    parser.add_argument("--eval-every", type=int, default=250, help="Training images between evaluations")
    parser.add_argument("--patience", type=int, default=3, help="Evaluations without improvement before stopping")
    parser.add_argument(
        "--workers", type=int, default=1, help="Each worker costs ~1.5GB RAM and queues HD images in /dev/shm"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    teacher = load_model(args.arch, args.weights).cuda().eval().requires_grad_(False)
    config = foveation_config(vars(args))
    student = make_student(teacher, config, args.rank)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(f"Training {trainable:,} LoRA parameters on {config}")
    (args.out / "config.json").write_text(json.dumps(vars(args) | dict(trainable=trainable), default=str, indent=2))

    exclude = frozenset(p.name for p in args.exclude.iterdir()) if args.exclude else frozenset()
    train = Images(args.train, args.sizes, teacher.patch_size, augment=True, exclude=exclude)
    val = Images(args.val, args.sizes, teacher.patch_size, augment=False, limit=args.val_images, exclude=exclude)
    logger.info(f"{len(train)} training and {len(val)} held-out images")
    loader = DataLoader(train, batch_size=None, shuffle=True, num_workers=args.workers, persistent_workers=True)
    steps = args.images // args.accumulate
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)),
    )

    def held_out_loss(scores: dict[str, float]) -> float:
        return 2 - scores["cls_cosine"] - scores["patch_cosine"]

    def configure(ratio: float, image: Tensor) -> None:
        H, W = (side // teacher.patch_size for side in image.shape[-2:])
        reconfigure(student, budget_config(config, H, W, ratio, 1 + teacher.n_storage_tokens))

    def configure_held_out(i: int, image: Tensor) -> None:
        if args.ratios:
            configure(held_out_ratio(i, args.ratios), image)

    def held_out() -> dict[str, float]:
        return evaluate(student, teacher, val, configure_held_out)

    history = [dict(images=0, loss=float("nan")) | held_out()]
    logger.info(f"Before training: {history[-1]}")
    best, since_best = held_out_loss(history[-1]), 0
    torch.save(get_peft_model_state_dict(student), args.out / "adapter.pt")
    seen, running, start = 0, [], time.time()
    while seen < args.images and since_best < args.patience:
        for image in loader:
            image = image[None].cuda(non_blocking=True)
            if args.ratios:
                configure(math.exp(random.uniform(*map(math.log, args.ratios))), image)
            with torch.no_grad():
                target = features(teacher, image)
            cls_distance, patch_distance = distance(features(student, image), target)
            loss = cls_distance + patch_distance
            (loss / args.accumulate).backward()
            running.append(loss.item())
            seen += 1
            if seen % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            if seen % 100 == 0:
                rate = seen / (time.time() - start)
                logger.info(f"{seen} images, loss {sum(running) / len(running):.4f}, {rate:.1f} images/s")
            if seen % args.eval_every == 0 or seen == args.images:
                history.append(dict(images=seen, loss=sum(running) / len(running)) | held_out())
                running = []
                pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
                if held_out_loss(history[-1]) < best:  # keeps the best adapter, so stopping late costs nothing
                    best, since_best = held_out_loss(history[-1]), 0
                    torch.save(get_peft_model_state_dict(student), args.out / "adapter.pt")
                else:
                    since_best += 1
                logger.info(f"Eval: {history[-1]}, {since_best} since the best")
            if seen == args.images or since_best == args.patience:
                break

    logger.info(f"Stopped after {seen} images; best adapter and history in {args.out}")


if __name__ == "__main__":
    main()
