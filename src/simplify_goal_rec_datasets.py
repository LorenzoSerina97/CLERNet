"""Convert pickle plan files into compact JSON training arrays.

This is the second step in the data-preparation pipeline:

    1. src/data/dataset.py           — raw .SOL files → pickle of PlanOnline objects
    2. **simplify_goal_rec_datasets.py** (this script) — pickle → compact JSON arrays

The output JSON stores (action_ids, goal_indices) pairs that can be loaded
directly by the training CLI without re-running the generator at every epoch.

Usage example (blocksworld):

    python src/simplify_goal_rec_datasets.py \\
        --domain      blocksworld \\
        --read-dict-dir  data/custom_dataset/blocksworld \\
        --read-plans-dir data/training_data/blocksworld/example_raw_training_data/plans_pickle \\
        --out-dir     data/training_data/blocksworld/compact_training_data
"""

import json
import random

import click

from src.constants import DEFAULTS, FILENAMES, HELPS, MAX_PLAN_DIMS
from src.data.plan_generator import PlanGeneratorOnlineSimple
from src.utils import load_from_pickles


@click.command()
@click.option(
    "--domain",
    "domain_name",
    required=True,
    prompt=True,
    type=click.Choice(list(MAX_PLAN_DIMS.keys())),
    help=HELPS.DOMAIN_NAME,
)
@click.option(
    "--read-dict-dir",
    "read_dict_dir",
    required=True,
    prompt=True,
    type=click.Path(exists=True, file_okay=False),
    help=HELPS.DICT_FOLDER_SRC + " (contains 'dizionario' and 'dizionario_goal' pickles)",
)
@click.option(
    "--read-plans-dir",
    "read_plans_dir",
    required=True,
    prompt=True,
    type=click.Path(exists=True, file_okay=False),
    help=HELPS.PLANS_FOLDER_SRC + " (contains train_plans and val_plans pickles)",
)
@click.option(
    "--out-dir",
    "out_dir",
    required=True,
    prompt=True,
    type=click.STRING,
    help="Directory where the compact JSON files will be saved.",
)
@click.option(
    "--max-val-samples",
    "max_val_samples",
    default=5500,
    type=click.INT,
    show_default=True,
    help="Maximum number of validation samples to keep.",
)
def simplify_goal_rec_datasets(
    domain_name: str,
    read_dict_dir: str,
    read_plans_dir: str,
    out_dir: str,
    max_val_samples: int,
) -> None:
    """Build compact JSON datasets from pickled PlanOnline objects."""
    import os

    max_plan_dim = MAX_PLAN_DIMS[domain_name]

    action_vocab, goal_vocab = load_from_pickles(
        read_dict_dir, [FILENAMES.ACTION_DICT_FILENAME, FILENAMES.GOALS_DICT_FILENAME]
    )

    train_plans, val_plans = load_from_pickles(
        read_plans_dir,
        [FILENAMES.TRAIN_PLANS_FILENAME, FILENAMES.VALIDATION_PLANS_FILENAME],
    )

    random.shuffle(val_plans)
    val_plans = val_plans[:max_val_samples]

    train_generator = PlanGeneratorOnlineSimple(
        train_plans,
        action_vocab,
        goal_vocab,
        len(train_plans),
        max_dim=max_plan_dim,
        shuffle=False,
        truncate=True,
        zero_padding=True,
    )
    val_generator = PlanGeneratorOnlineSimple(
        val_plans,
        action_vocab,
        goal_vocab,
        len(val_plans),
        max_dim=max_plan_dim,
        shuffle=False,
        truncate=True,
        zero_padding=True,
    )

    x_train, y_train = train_generator[0]
    train_simple = list(zip(x_train, y_train))
    print(f"Sample train entry: {train_simple[0]}")

    x_val, y_val = val_generator[0]
    val_simple = list(zip(x_val, y_val))

    os.makedirs(out_dir, exist_ok=True)

    train_out = os.path.join(out_dir, f"compact_{FILENAMES.TRAIN_PLANS_FILENAME_SIMPLE_JSON}")
    val_out = os.path.join(out_dir, f"compact_{FILENAMES.VALIDATION_PLANS_FILENAME_SIMPLE_JSON}")

    with open(train_out, "w") as f:
        json.dump(train_simple, f)
    with open(val_out, "w") as f:
        json.dump(val_simple, f)

    print(f"Saved {len(train_simple)} training samples → {train_out}")
    print(f"Saved {len(val_simple)} validation samples → {val_out}")


if __name__ == "__main__":
    simplify_goal_rec_datasets()
