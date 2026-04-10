# CLERNet Goal Recognition

CLERNet is an online goal recognition framework that uses LSTM/GRU networks to predict an agent's goal from a partial sequence of observed actions. Given an incomplete plan execution, the model outputs a probability over all candidate goals at each observation step.

Six pre-trained models are included: **blocksworld**, **depots**, **driverlog**, **logistics**, **satellite**, and **zenotravel**.

---

## Project Structure

```
CLERNet_public/
├── data/
│   ├── custom_dataset/          # Test instances per domain (ZIP files + dictionaries)
│   │   └── {domain}/
│   │       ├── dizionario        # Action vocabulary (pickle)
│   │       ├── dizionario_goal   # Goal vocabulary (pickle)
│   │       └── 100/              # ZIP test instances (obs.dat, real_hyp.dat, ...)
│   ├── params_optuna/           # Best Optuna hyperparameter JSON files per domain
│   └── training_data/           # Training data (excluded from git — see below)
│       └── {domain}/
│           ├── example_raw_training_data/   # Example .SOL files
│           └── compact_training_data/       # Compact JSON arrays for training
├── models/                      # Pre-trained .keras model files (one per domain)
├── src/
│   ├── cli/
│   │   └── test_model.py        # clernet-test CLI — evaluate a model on test instances
│   ├── data/
│   │   ├── action.py            # Action representation
│   │   ├── plan.py              # Plan and PlanOnline representations
│   │   ├── plan_generator.py    # Keras Sequence generators for training
│   │   └── dataset.py           # Step 1: .SOL files → pickle of PlanOnline objects
│   ├── models/
│   │   ├── architecture.py      # Network builder (build_network_single_fact)
│   │   ├── attention.py         # Custom attention layers
│   │   ├── loss.py              # Custom loss functions
│   │   └── params_generator.py  # Hyperparameter grid generator
│   ├── training/
│   │   └── train.py             # clernet-train CLI — train and Optuna search
│   ├── constants.py             # Domain constants, filenames, defaults
│   ├── simplify_goal_rec_datasets.py  # Step 2: pickle → compact JSON arrays
│   └── utils.py                 # I/O helpers
└── pyproject.toml
```

---

## Installation

Python 3.8+ required. Using a virtual environment is recommended.

```bash
pip install -e .
```

This installs two CLI entry points:
- `clernet-train` — train new models or run Optuna hyperparameter search
- `clernet-test` — evaluate a pre-trained model on test instances

---

## Data Pipeline

Training data is **not tracked in git** (large files). The pipeline has two steps:

### Step 1 — `.SOL` files to pickle

```bash
python -m src.data.dataset \
    --sol-dir  data/training_data/blocksworld/example_raw_training_data \
    --out-dir  data/training_data/blocksworld/plans_pickle \
    --domain   blocksworld \
    --split    0.7
```

Produces `train_plans` and `val_plans` pickle files alongside the existing `dizionario` and `dizionario_goal` vocabulary pickles.

### Step 2 — Pickle to compact JSON

```bash
python -m src.simplify_goal_rec_datasets \
    --domain         blocksworld \
    --read-dict-dir  data/custom_dataset/blocksworld \
    --read-plans-dir data/training_data/blocksworld/plans_pickle \
    --out-dir        data/training_data/blocksworld/compact_training_data
```

Produces `compact_train_plans_simple.json` and `compact_val_lpg_plans_simple.json`, which the training CLI reads directly.

---

## Training

### Single training run

```bash
clernet-train train-model \
    --target-dir   ./models \
    --plan-perc    1.0 \
    --batch-size   64 \
    --read-dict-dir ./data/custom_dataset/blocksworld \
    --epochs       50 \
    --read-plans-dir ./data/training_data/blocksworld/compact_training_data \
    --max-plan-dim 75 \
    neural-network \
    --network-params ./data/params_optuna/blocksworld/blocksworld_params.json \
    train
```

Add `--max-train-samples N` / `--max-val-samples N` to limit dataset size for quick smoke tests:

```bash
    train --max-train-samples 500 --max-val-samples 100
```

### Optuna hyperparameter search

```bash
clernet-train train-model \
    --target-dir   ./models \
    --plan-perc    1.0 \
    --batch-size   32 \
    --read-dict-dir ./data/custom_dataset/blocksworld \
    --epochs       50 \
    --read-plans-dir ./data/training_data/blocksworld/compact_training_data \
    --max-plan-dim 75 \
    optuna \
    --model-name blocksworld \
    --trials     50 \
    --db-dir     ./optuna_db
```

Once the study is complete, export the best parameters to JSON:

```bash
clernet-train optuna-create-best-params-json \
    --model-name blocksworld \
    --db-dir     ./optuna_db \
    --out-dir    ./data/params_optuna/blocksworld
```

**Supported domains and their sequence lengths (`--max-plan-dim`):**

| Domain | Max plan dim |
|--------|-------------|
| blocksworld | 75 |
| depots | 50 |
| driverlog | 70 |
| logistics | 50 |
| satellite | 40 |
| zenotravel | 40 |

---

## Evaluating Pre-trained Models

```bash
clernet-test \
    --model-path  ./models/blocksworld.keras \
    --domain      blocksworld \
    --data-dir    ./data/custom_dataset \
    --n-instances 10
```

Optional: save results to a file with `--output-dir ./results`.

Each test instance is a ZIP file under `{data-dir}/{domain}/100/` containing:
- `obs.dat` — observed actions (one per line)
- `real_hyp.dat` — true goal predicates
- `hyps.dat` — all candidate hypotheses
- `domain.pddl` / `template.pddl` — PDDL files

**Example output (blocksworld, 10 instances):**

```
[ 1/10] blocksworld_p000002_hyp=hyp-6_100.zip      F1=0.538  Acc=0.989  Ham=-0.011
[ 2/10] blocksworld_p000012_hyp=hyp-15_100.zip     F1=0.612  Acc=0.987  Ham=-0.013
...
=== Summary over 10 instances ===
  Weighted avg F1:           0.5731
  Weighted avg Accuracy:     0.9885
  Weighted avg Hamming Loss: -0.0115
```

Metrics are step-weighted: later observations (with more context) contribute more to the final score.

---

## Network Architecture

CLERNet uses a sequential architecture:

```
Input (action IDs, shape: seq_len)
  → Embedding (action_vocab_size → embedding_dim, mask_zero=True)
  → LSTM (units, return_sequences=True)
  → [optional] Attention
  → TimeDistributed(Dense(n_goals, activation='sigmoid'))
Output (goal probabilities at each step, shape: seq_len × n_goals)
```

At inference time, predictions are accumulated step-by-step: the running sum of outputs is thresholded at 0.5 to produce the current goal prediction. This makes the model's confidence grow as more actions are observed.

**Supported loss functions:** `binary_crossentropy`, `binary_focal_crossentropy`, and several custom variants (`rmse`, `bce_ol`, `bce_op`, `bce_hmlp`, `bfce_ol`, ...).

---

## Citation

If you use CLERNet in your research, please cite:

```bibtex
@inproceedings{DBLP:conf/ifaamas/SerinaCGPS25,
  author       = {Lorenzo Serina and
                  Mattia Chiari and
                  Alfonso Emilio Gerevini and
                  Luca Putelli and
                  Ivan Serina},
  title        = {Towards Efficient Online Goal Recognition through Deep Learning},
  booktitle    = {Proceedings of the 24th International Conference on Autonomous Agents
                  and Multiagent Systems, {AAMAS} 2025, Detroit, MI, USA, May 19-23,
                  2025},
  pages        = {1895--1903},
  publisher    = {International Foundation for Autonomous Agents and Multiagent Systems
                  / {ACM}},
  year         = {2025},
  url          = {https://dl.acm.org/doi/10.5555/3709347.3743826},
  doi          = {10.5555/3709347.3743826},
}
```

---

## Licensing and Credits

Adapted and refactored from multiple upstream works to establish a clearer baseline for online Goal Recognition tasks.
