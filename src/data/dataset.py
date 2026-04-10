"""Convert raw .SOL plan files into a pickle of PlanOnline objects.

This is the first step in the data-preparation pipeline:

    1. **dataset.py** (this script) — raw .SOL files → pickle of PlanOnline objects
    2. simplify_goal_rec_datasets.py  — pickle → compact JSON training arrays

Usage example (blocksworld example data included in the repo):

    python -m src.data.dataset \\
        --sol-dir  data/training_data/blocksworld/example_raw_training_data \\
        --out-dir  data/training_data/blocksworld/example_raw_training_data/plans_pickle \\
        --split train
"""

import os
import pickle

import click

import src.data.plan as plan
from src.constants import FILENAMES, MAX_PLAN_DIMS


@click.command()
@click.option(
    "--sol-dir",
    "sol_dir",
    required=True,
    prompt=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing .SOL plan files.",
)
@click.option(
    "--out-dir",
    "out_dir",
    required=True,
    prompt=True,
    type=click.STRING,
    help="Directory where the pickle file will be saved.",
)
@click.option(
    "--split",
    "split",
    default="train",
    type=click.Choice(["train", "val", "test"]),
    show_default=True,
    help="Dataset split label used to name the output pickle file.",
)
def create_plans_pickle(sol_dir: str, out_dir: str, split: str) -> None:
    """Parse all .SOL files in SOL_DIR and save them as a pickle."""
    plans = []
    skipped = 0

    for sol_file in sorted(os.listdir(sol_dir)):
        if not sol_file.endswith(".SOL"):
            continue
        sol_path = os.path.join(sol_dir, sol_file)
        try:
            parsed_plan = plan.PlanOnline(sol_path)
            plans.append(parsed_plan)
        except Exception as exc:
            print(f"[WARN] Could not parse {sol_path}: {exc}")
            skipped += 1

    print(f"Parsed {len(plans)} plans ({skipped} skipped) from {sol_dir}")

    os.makedirs(out_dir, exist_ok=True)

    filename_map = {
        "train": FILENAMES.TRAIN_PLANS_FILENAME,
        "val": FILENAMES.VALIDATION_PLANS_FILENAME,
        "test": FILENAMES.TEST_PLANS_FILENAME,
    }
    out_path = os.path.join(out_dir, filename_map[split])
    with open(out_path, "wb") as f:
        pickle.dump(plans, f)

    print(f"Saved pickle to {out_path}")


if __name__ == "__main__":
    create_plans_pickle()
