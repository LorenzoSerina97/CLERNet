"""Evaluate a pre-trained CLERNet model on goal recognition instances.

Each test instance is stored as a ZIP file in ``{data_dir}/{domain}/100/``
and contains:
- ``obs.dat``      — observed actions (one per line, e.g. ``(UNSTACK B H)``)
- ``real_hyp.dat`` — true goal predicates (e.g. ``(ON F A), (ON K F), ...``)
- ``hyps.dat``     — all candidate goal hypotheses
- ``domain.pddl``  — PDDL domain definition
- ``template.pddl``— PDDL problem template

Example usage::

    clernet-test \\
        --model-path ./models/blocksworld.keras \\
        --domain blocksworld \\
        --data-dir ./data/custom_dataset \\
        --n-instances 10
"""

import h5py
import json
import os
import shutil
import tempfile
import zipfile
from os.path import join
from typing import Optional, Tuple

import click
import keras
import numpy as np
from keras.models import load_model
from sklearn import metrics as sk_metrics

from src.constants import FILENAMES, MAX_PLAN_DIMS
from src.models.architecture import CUSTOM_OBJECTS
from src.utils import load_from_pickles


# ---------------------------------------------------------------------------
# Legacy model loader (Keras 2 HDF5 → Keras 3)
# ---------------------------------------------------------------------------

_LEGACY_RNN_DISCARD = {"time_major", "implementation"}

_LAYER_BUILDERS = {
    "Embedding": lambda c: keras.layers.Embedding(
        input_dim=c["input_dim"],
        output_dim=c["output_dim"],
        embeddings_initializer=c.get("embeddings_initializer", "uniform"),
        mask_zero=c.get("mask_zero", False),
        name=c["name"],
    ),
    "LSTM": lambda c: keras.layers.LSTM(
        **{k: v for k, v in c.items()
           if k not in _LEGACY_RNN_DISCARD
           and k not in ("name", "trainable", "dtype", "batch_input_shape",
                         "go_backwards", "stateful", "unroll",
                         "unit_forget_bias", "kernel_initializer",
                         "recurrent_initializer", "bias_initializer",
                         "kernel_regularizer", "recurrent_regularizer",
                         "bias_regularizer", "activity_regularizer",
                         "kernel_constraint", "recurrent_constraint",
                         "bias_constraint", "recurrent_activation")},
        name=c["name"],
    ),
    "GRU": lambda c: keras.layers.GRU(
        **{k: v for k, v in c.items()
           if k not in _LEGACY_RNN_DISCARD
           and k not in ("name", "trainable", "dtype", "batch_input_shape",
                         "go_backwards", "stateful", "unroll",
                         "reset_after", "kernel_initializer",
                         "recurrent_initializer", "bias_initializer",
                         "kernel_regularizer", "recurrent_regularizer",
                         "bias_regularizer", "activity_regularizer",
                         "kernel_constraint", "recurrent_constraint",
                         "bias_constraint", "recurrent_activation")},
        name=c["name"],
    ),
    "Dense": lambda c: keras.layers.Dense(
        units=c["units"],
        activation=c.get("activation", None),
        name=c["name"],
    ),
    "TimeDistributed": None,  # handled specially below
    "Dropout": lambda c: keras.layers.Dropout(rate=c["rate"], name=c["name"]),
}


def _build_from_keras2_config(layers_cfg: list, custom_objects: dict):
    """Reconstruct a Sequential-style Functional model from Keras 2 layer list."""
    # Skip InputLayer — we derive input shape from the next layer's config
    seq = [l for l in layers_cfg if l["class_name"] != "InputLayer"]

    # Determine input shape from the embedding's batch_input_shape
    first_cfg = layers_cfg[0]["config"]
    if "batch_input_shape" in first_cfg:
        input_shape = tuple(first_cfg["batch_input_shape"][1:])
    else:
        input_shape = tuple(layers_cfg[1]["config"].get("batch_input_shape", [None])[1:])

    inputs = keras.Input(shape=input_shape, dtype="int32")
    x = inputs
    for layer_def in seq:
        cls = layer_def["class_name"]
        cfg = layer_def["config"]
        if cls == "TimeDistributed":
            inner_cfg = cfg["layer"]["config"]
            inner_cls = cfg["layer"]["class_name"]
            inner_layer = _LAYER_BUILDERS[inner_cls](inner_cfg)
            x = keras.layers.TimeDistributed(inner_layer, name=cfg["name"])(x)
        elif cls in _LAYER_BUILDERS:
            x = _LAYER_BUILDERS[cls](cfg)(x)
        elif cls in custom_objects:
            # custom layer: pass all config keys
            x = custom_objects[cls](**{k: v for k, v in cfg.items()
                                       if k not in ("name", "trainable", "dtype")},
                                    name=cfg["name"])(x)
        else:
            raise ValueError(f"Unsupported layer class: {cls}")

    return keras.Model(inputs, x)


def _load_model(model_path: str, custom_objects: dict):
    """Load a Keras model, handling both Keras 3 zip format and legacy HDF5.

    Pre-trained CLERNet models were saved with Keras 2.4 as HDF5 files.
    Keras 3 expects ``.keras`` files to be zip archives, so we detect the
    format via ``h5py`` and fall back to weight-loading when necessary.
    """
    try:
        # Standard path — works for Keras-3-format .keras zip files
        return load_model(model_path, custom_objects=custom_objects)
    except Exception:
        pass

    # Legacy HDF5 path
    with h5py.File(model_path, "r") as f:
        config_raw = f.attrs.get("model_config", None)
        if config_raw is None:
            raise ValueError(f"Cannot determine model config from {model_path}")
        if isinstance(config_raw, bytes):
            config_raw = config_raw.decode("utf-8")
        model_config = json.loads(config_raw)

    layers_cfg = model_config.get("config", {}).get("layers", [])
    model = _build_from_keras2_config(layers_cfg, custom_objects)

    # Load weights via a temporary .h5 file (Keras 3 infers format from ext)
    tmp_path = tempfile.mktemp(suffix=".h5")
    try:
        shutil.copy(model_path, tmp_path)
        model.load_weights(tmp_path, by_name=True)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return model


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_obs(obs_text: str) -> list:
    """Parse obs.dat content into a list of action name strings.

    Example input line: ``(UNSTACK B H)``
    Example output: ``["UNSTACK B H", ...]``
    """
    actions = []
    for line in obs_text.strip().splitlines():
        line = line.strip()
        if line:
            actions.append(line.strip("()").strip())
    return actions


def _parse_hyps(hyps_text: str, goal_vocab: dict) -> list:
    """Parse hyps.dat into a list of candidate hypotheses.

    Each line of ``hyps.dat`` is one hypothesis with comma-separated predicates,
    e.g. ``(ON A K), (ON B H), ...``.

    Args:
        hyps_text: Raw text content of ``hyps.dat``.
        goal_vocab: Predicate string → one-hot vector mapping.

    Returns:
        List of hypotheses; each hypothesis is a sorted list of goal vocab indices.
    """
    hypotheses = []
    for line in hyps_text.strip().splitlines():
        predicates = [
            token.strip().strip("()").strip().lower()
            for token in line.split(",")
        ]
        mask = _goal_mask([p for p in predicates if p], goal_vocab)
        indices = sorted(np.where(mask > 0)[0].tolist())
        if indices:
            hypotheses.append(indices)
    return hypotheses


def _find_correct_hyp_idx(goal_vector: np.ndarray, possible_goals: list) -> int:
    """Return the index of the hypothesis that matches the true goal vector.

    Args:
        goal_vector: Binary array of shape ``(n_goals,)`` for the true goal.
        possible_goals: List of hypotheses, each a list of goal vocab indices.

    Returns:
        Index into ``possible_goals``, or ``-1`` if no match found.
    """
    correct_set = set(np.where(goal_vector > 0)[0].tolist())
    for i, hyp in enumerate(possible_goals):
        if set(hyp) == correct_set:
            return i
    return -1


def _parse_real_hyp(real_hyp_text: str) -> list:
    """Parse real_hyp.dat content into lowercase predicate strings.

    Example input: ``(ON F A), (ON K F), (ON H K)``
    Example output: ``["on f a", "on k f", "on h k"]``
    """
    predicates = []
    for token in real_hyp_text.strip().split(","):
        pred = token.strip().strip("()").strip().lower()
        if pred:
            predicates.append(pred)
    return predicates


def _goal_mask(predicates: list, goal_vocab: dict) -> np.ndarray:
    """Convert goal predicate strings to a binary multi-hot goal vector.

    Args:
        predicates: Lowercase predicate strings, e.g. ``["on f a", "on k f"]``.
        goal_vocab: Dict mapping predicate string → one-hot numpy vector.

    Returns:
        Binary array of shape ``(n_goals,)``.
    """
    n_goals = len(goal_vocab)
    mask = np.zeros(n_goals, dtype=np.float32)
    for pred in predicates:
        if pred in goal_vocab:
            mask += np.array(goal_vocab[pred], dtype=np.float32)
        else:
            print(f"[WARN] Goal predicate '{pred}' not found in goal_vocab")
    return (mask > 0).astype(np.float32)


def _encode_actions(actions: list, action_vocab: dict, max_plan_dim: int) -> np.ndarray:
    """Encode action names as integer IDs; truncate/pad to ``max_plan_dim``.

    Unknown actions are mapped to 0 (the pad token).

    Args:
        actions: List of action name strings, e.g. ``["UNSTACK B H", ...]``.
        action_vocab: Dict mapping action name → integer ID.
        max_plan_dim: Fixed sequence length for the model.

    Returns:
        Integer array of shape ``(max_plan_dim,)``.
    """
    ids = [action_vocab.get(a, 0) for a in actions[:max_plan_dim]]
    ids += [0] * (max_plan_dim - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    zip_path: str,
    model,
    action_vocab: dict,
    goal_vocab: dict,
    max_plan_dim: int,
) -> Tuple:
    """Run the model on a single ZIP test instance.

    Args:
        zip_path: Path to the ZIP file.
        model: Loaded Keras model.
        action_vocab: Action name → integer ID mapping.
        goal_vocab: Predicate string → one-hot vector mapping.
        max_plan_dim: Sequence length expected by the model.

    Returns:
        Tuple ``(y_pred_steps, y_true_steps, possible_goals, correct_hyp_idx)``
        where ``y_pred_steps`` and ``y_true_steps`` have shape ``(obs_len, n_goals)``,
        ``possible_goals`` is a list of hypothesis index lists parsed from ``hyps.dat``,
        and ``correct_hyp_idx`` is the index of the true goal in ``possible_goals``
        (or ``-1`` if not found).  Returns ``(None, None, None, None)`` on parse failure.
    """
    with zipfile.ZipFile(zip_path) as z:
        obs_text = z.read("obs.dat").decode("utf-8")
        real_hyp_text = z.read("real_hyp.dat").decode("utf-8")
        hyps_text = z.read("hyps.dat").decode("utf-8")

    actions = _parse_obs(obs_text)
    if not actions:
        return None, None, None, None

    obs_len = min(len(actions), max_plan_dim)
    x = _encode_actions(actions, action_vocab, max_plan_dim)

    goal_predicates = _parse_real_hyp(real_hyp_text)
    goal_vector = _goal_mask(goal_predicates, goal_vocab)

    possible_goals = _parse_hyps(hyps_text, goal_vocab)
    correct_hyp_idx = _find_correct_hyp_idx(goal_vector, possible_goals)

    # Predict: input shape (1, max_plan_dim), output (1, max_plan_dim, n_goals)
    y_pred = model.predict(x[np.newaxis, :], verbose=0)[0]  # (max_plan_dim, n_goals)

    # Ground truth: same goal vector at every observed step
    y_true = np.tile(goal_vector, (obs_len, 1))  # (obs_len, n_goals)

    return y_pred[:obs_len], y_true, possible_goals, correct_hyp_idx


def _compute_weighted_metrics(y_pred_steps: np.ndarray, y_true_steps: np.ndarray) -> dict:
    """Compute step-weighted F1, accuracy, and Hamming loss.

    Uses a cumulative prediction approach: at step t the running sum of all
    predictions so far is thresholded at 0.5.  Later steps receive higher
    weight (weight = step index, 1-indexed).

    Returns:
        Dict with keys ``f1``, ``accuracy``, ``hamming``.
    """
    actual_pred = np.zeros(y_pred_steps.shape[-1])
    f1_track, acc_track, ham_track = [], [], []

    for t in range(len(y_pred_steps)):
        actual_pred += y_pred_steps[t]
        current_pred = (actual_pred >= 0.5).astype(int).tolist()
        current_goal = (y_true_steps[t] > 0).astype(int).tolist()

        acc_track.append(sk_metrics.accuracy_score(current_goal, current_pred))
        f1_track.append(sk_metrics.f1_score(current_goal, current_pred, zero_division=0))
        ham_track.append(-sk_metrics.hamming_loss(current_goal, current_pred))

    weights = list(range(1, len(f1_track) + 1))
    w_sum = sum(weights)
    return {
        "f1": sum(f * w for f, w in zip(f1_track, weights)) / w_sum,
        "accuracy": sum(a * w for a, w in zip(acc_track, weights)) / w_sum,
        "hamming": sum(h * w for h, w in zip(ham_track, weights)) / w_sum,
    }


def _compute_rf_cv(
    y_pred_steps: np.ndarray,
    possible_goals: list,
    correct_hyp_idx: int,
) -> dict:
    """Compute RF (Ranked First) and CV (Convergence) for one instance.

    At each step the cumulative sum of predictions is used to score every
    candidate hypothesis; the hypothesis with the highest score is selected.

    - **RF**: fraction of steps where the selected hypothesis is the correct one.
    - **CV**: fraction of steps covered by the longest correct trailing suffix,
      i.e. ``(n - t*) / n`` where ``t*`` is the earliest step from which the
      model selects the correct hypothesis without ever switching away.

    Args:
        y_pred_steps: Raw per-fluent predictions, shape ``(obs_len, n_goals)``.
        possible_goals: List of hypotheses, each a list of goal vocab indices.
        correct_hyp_idx: Index of the true hypothesis in ``possible_goals``.

    Returns:
        Dict with keys ``rf`` and ``cv``.
    """
    running = np.zeros(y_pred_steps.shape[-1])
    step_correct = []

    for t in range(len(y_pred_steps)):
        running += y_pred_steps[t]
        scores = [float(np.sum(running[hyp])) for hyp in possible_goals]
        selected = int(np.argmax(scores))
        step_correct.append(1 if selected == correct_hyp_idx else 0)

    n = len(step_correct)
    rf = sum(step_correct) / n if n else 0.0

    # Find earliest step of the final correct streak
    t_star = n
    for t in range(n - 1, -1, -1):
        if step_correct[t] == 1:
            t_star = t
        else:
            break
    cv = (n - t_star) / n if n else 0.0

    return {"rf": rf, "cv": cv}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--model-path",
    "model_path",
    required=True,
    prompt=True,
    type=click.Path(exists=True),
    help="Path to the .keras model file.",
)
@click.option(
    "--domain",
    "domain_name",
    required=True,
    prompt=True,
    type=click.Choice(list(MAX_PLAN_DIMS.keys())),
    help="Planning domain name.",
)
@click.option(
    "--data-dir",
    "data_dir",
    required=True,
    prompt=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root of custom_dataset/ (e.g. ./data/custom_dataset).",
)
@click.option(
    "--n-instances",
    "n_instances",
    default=10,
    type=click.INT,
    show_default=True,
    help="Number of test instances to evaluate.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.STRING,
    help="Directory to write a metrics .txt file (optional).",
)
def run_test(
    model_path: str,
    domain_name: str,
    data_dir: str,
    n_instances: int,
    output_dir: Optional[str],
) -> None:
    """Evaluate a pre-trained CLERNet model on N goal recognition instances."""
    max_plan_dim = MAX_PLAN_DIMS[domain_name]
    dict_dir = join(data_dir, domain_name)
    zip_dir = join(data_dir, domain_name, "100")

    print(f"Loading dictionaries from {dict_dir} ...")
    [action_vocab, goal_vocab] = load_from_pickles(
        dict_dir, [FILENAMES.ACTION_DICT_FILENAME, FILENAMES.GOALS_DICT_FILENAME]
    )

    print(f"Loading model from {model_path} ...")
    model = _load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    zip_files = sorted(f for f in os.listdir(zip_dir) if f.endswith(".zip"))[:n_instances]

    if not zip_files:
        print(f"No ZIP files found in {zip_dir}")
        return

    print(f"Evaluating on {len(zip_files)} instance(s) ...\n")

    all_metrics = []
    for idx, zf in enumerate(zip_files, 1):
        zip_path = join(zip_dir, zf)
        y_pred_steps, y_true_steps, possible_goals, correct_hyp_idx = evaluate_instance(
            zip_path, model, action_vocab, goal_vocab, max_plan_dim
        )
        if y_pred_steps is None:
            print(f"[{idx}/{len(zip_files)}] {zf}: skipped (parse error)")
            continue

        m = _compute_weighted_metrics(y_pred_steps, y_true_steps)
        if possible_goals and correct_hyp_idx != -1:
            m.update(_compute_rf_cv(y_pred_steps, possible_goals, correct_hyp_idx))
        else:
            m["rf"] = float("nan")
            m["cv"] = float("nan")

        all_metrics.append(m)
        print(
            f"[{idx:2d}/{len(zip_files)}] {zf[:50]:<50}  "
            f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  Ham={m['hamming']:.3f}  "
            f"RF={m['rf']:.3f}  CV={m['cv']:.3f}"
        )

    if not all_metrics:
        print("No valid results.")
        return

    def _avg(key):
        vals = [m[key] for m in all_metrics if not (isinstance(m[key], float) and m[key] != m[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    summary_lines = [
        "",
        f"=== Summary over {len(all_metrics)} instances ===",
        f"  Weighted avg F1:           {_avg('f1'):.4f}",
        f"  Weighted avg Accuracy:     {_avg('accuracy'):.4f}",
        f"  Weighted avg Hamming Loss: {_avg('hamming'):.4f}",
        f"  Avg RF (Ranked First):     {_avg('rf'):.4f}",
        f"  Avg CV (Convergence):      {_avg('cv'):.4f}",
    ]
    for line in summary_lines:
        print(line)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = join(output_dir, f"{domain_name}_test_metrics.txt")
        with open(out_path, "w") as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_test()
