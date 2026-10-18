"""Phase 1: distil dense DINOv3 into a foveated student with LoRA adapters on the attention projections.

Teacher: the frozen dense model. Student: the same weights, foveated, plus LoRA adapters on every block's qkv and
proj (via peft); everything else stays frozen. Loss: 1 - cosine similarity between student and teacher final
normalised features, for the CLS token plus the mean over patch tokens. Gradients reach the adapters through the
pooled keys and values and the final attention, but not through the layout choice.

Every --eval-every images, student and teacher are compared end to end on held-out images, starting with the
untrained student at step 0. Adapters and metrics go to --out.

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
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dinov3.data.transforms import make_base_transform
from dinov3.foveation.attention import FoveationConfig, foveate
from dinov3.foveation.phase0 import ARCHS, IMAGE_SUFFIXES, load_model, resize_image

logger = logging.getLogger("dinov3")


class Images(Dataset):
    """uint8 images resized to a long side of `size` pixels, optionally flipped at random."""

    def __init__(self, folder: Path, size: int, patch_size: int, flip: bool, limit: int | None = None):
        self.paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]
        self.size, self.patch_size, self.flip = size, patch_size, flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tensor:
        image = resize_image(self.paths[i], self.size, self.patch_size)
        return image.flip(-1) if self.flip and random.random() < 0.5 else image


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
    )


def make_student(teacher, config: FoveationConfig, rank: int):
    """A foveated copy of the teacher with LoRA adapters on every block's qkv and proj."""
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=r"blocks\.\d+\.attn\.(qkv|proj)")
    return get_peft_model(foveate(copy.deepcopy(teacher), config), lora)


@torch.no_grad()
def evaluate(student, teacher, images: Images) -> dict[str, float]:
    cls, patches = [], []
    for image in images:
        image = image[None].cuda()
        cls_distance, patch_distance = distance(features(student, image), features(teacher, image))
        cls.append(1 - cls_distance.item())
        patches.append(1 - patch_distance.item())
    return dict(cls_cosine=sum(cls) / len(cls), patch_cosine=sum(patches) / len(patches))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, default="vits16")
    parser.add_argument("--weights", required=True, help="Hugging Face model directory, or a Meta .pth path or URL")
    parser.add_argument("--train", type=Path, required=True, help="Directory of training images")
    parser.add_argument("--val", type=Path, required=True, help="Directory of held-out images")
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--size", type=int, default=1920, help="Long side in pixels")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--start-level", type=int, default=2)
    parser.add_argument("--order", choices=["per_level", "global", "tree"], default="tree")
    parser.add_argument("--score", choices=["attention", "detail", "bound", "random"], default="bound")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--images", type=int, default=4000, help="Training images seen in total")
    parser.add_argument("--accumulate", type=int, default=4, help="Images per optimiser step")
    parser.add_argument("--warmup", type=int, default=50, help="Optimiser steps of linear warm-up")
    parser.add_argument("--eval-every", type=int, default=1000, help="Training images between evaluations")
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

    train = Images(args.train, args.size, teacher.patch_size, flip=True)
    val = Images(args.val, args.size, teacher.patch_size, flip=False, limit=args.val_images)
    loader = DataLoader(train, batch_size=None, shuffle=True, num_workers=args.workers, persistent_workers=True)
    steps = args.images // args.accumulate
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)),
    )

    history = [dict(images=0, loss=float("nan")) | evaluate(student, teacher, val)]
    logger.info(f"Before training: {history[-1]}")
    seen, running, start = 0, [], time.time()
    while seen < args.images:
        for image in loader:
            image = image[None].cuda(non_blocking=True)
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
                history.append(dict(images=seen, loss=sum(running) / len(running)) | evaluate(student, teacher, val))
                running = []
                logger.info(f"Eval: {history[-1]}")
                pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
            if seen == args.images:
                break

    torch.save(get_peft_model_state_dict(student), args.out / "adapter.pt")
    (args.out / "config.json").write_text(json.dumps(vars(args) | dict(trainable=trainable), default=str, indent=2))
    logger.info(f"Saved adapters and history to {args.out}")


if __name__ == "__main__":
    main()
