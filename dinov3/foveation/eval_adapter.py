"""Score a Phase 1 run's LoRA adapters against the dense teacher at several key budgets.

Each budget is scored with and without the adapters, so the gain from training shows separately from the cost of
the smaller or larger budget. The architecture, layout and image size come from the run's config.json.

Example:
    python -m dinov3.foveation.eval_adapter --run data/runs/pilot_vits16 --val data/div2k/DIV2K_valid_HR \\
        --budgets 1024 1536 2048
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from peft import set_peft_model_state_dict

from dinov3.foveation.distill import Images, evaluate, foveation_config, make_student
from dinov3.foveation.phase0 import load_model

logger = logging.getLogger("dinov3")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Output directory of a distill.py run")
    parser.add_argument("--val", type=Path, required=True, help="Directory of held-out images")
    parser.add_argument("--val-images", type=int, default=None, help="Default: all")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 1536, 2048])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    run = json.loads((args.run / "config.json").read_text())
    adapter = torch.load(args.run / "adapter.pt")
    teacher = load_model(run["arch"], run["weights"]).cuda().eval().requires_grad_(False)
    val = Images(args.val, run.get("sizes") or [run["size"]], teacher.patch_size, augment=False, limit=args.val_images)

    rows = []
    for budget in args.budgets:
        student = make_student(teacher, foveation_config(run | dict(budget=budget)), run["rank"]).eval()
        rows.append(dict(budget=budget, adapters=False) | evaluate(student, teacher, val))
        set_peft_model_state_dict(student, adapter)
        rows.append(dict(budget=budget, adapters=True) | evaluate(student, teacher, val))
        logger.info(f"Budget {budget}: {rows[-2]} -> {rows[-1]}")
        del student

    table = pd.DataFrame(rows)
    table.to_csv(args.run / "budgets.csv", index=False)
    print(table.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
