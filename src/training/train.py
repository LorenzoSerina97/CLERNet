import json
import os
import subprocess

import click
import numpy as np
import optuna
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from optuna.samplers import TPESampler
from optuna_integration.tfkeras import TFKerasPruningCallback
from os.path import join
from sklearn import metrics as sk_metrics

from src.constants import (
    DEFAULTS,
    ERRORS,
    FILENAMES,
    HELPS,
    KEYS,
    LOSSES_SHORT_NAMES,
    RANDOM_SEED,
)
from src.models.architecture import (
    CUSTOM_OBJECTS,
    build_network_single_fact,
    create_model_dir_name,
    get_callback_default_params,
    get_model_predictions,
    print_metrics,
    print_network_details,
    save_all_history_plots,
)
from src.models.params_generator import ParamsGenerator
from src.utils import load_file, load_from_pickles


# ---------------------------------------------------------------------------
# Optuna warm-up callback
# ---------------------------------------------------------------------------

class TFKerasPruningCallbackWarmup(TFKerasPruningCallback):
    """Extends Optuna's Keras pruning callback with a warmup period.

    During the first ``warmup_epochs`` epochs pruning is disabled so that
    trials are not cut off before the model has had a chance to learn.
    """

    def __init__(self, trial: optuna.Trial, monitor: str, warmup_epochs: int = 0):
        super().__init__(trial, monitor)
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self._warmup_epochs = warmup_epochs

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Skip pruning during the warmup period; delegate to parent afterwards."""
        if epoch >= self._warmup_epochs:
            super().on_epoch_end(epoch, logs)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _prepare_xy(plans: list, n_goals: int) -> tuple:
    """Unpack a list of (x, y) plan tuples into numpy arrays.

    The compact training format stores active goal *indices* per time step
    as a ragged list (e.g. ``[[113], [113], [], [113, 42], ...]``). This
    function converts them directly to a dense one-hot array without creating
    an inhomogeneous intermediate array.

    Args:
        plans: List of ``[action_ids, goal_index_sequence]`` pairs.
        n_goals: Size of the goal vocabulary (output dimension).

    Returns:
        Tuple ``(x, y)`` where x has shape ``(n, seq_len)`` and y has shape
        ``(n, seq_len, n_goals)``, both ready for model.fit / model.predict.
    """
    x_list, y_list = zip(*plans)
    x = np.array(x_list, dtype=np.int32)

    n_samples = len(y_list)
    seq_len = len(y_list[0])
    y_oh = np.zeros((n_samples, seq_len, n_goals), dtype=np.float32)

    for i, plan_goals in enumerate(y_list):
        for t, goal_indices in enumerate(plan_goals):
            for goal_idx in goal_indices:
                idx = int(goal_idx)
                if 0 <= idx < n_goals:
                    y_oh[i, t, idx] = 1.0

    return x, y_oh


def objective(
    trial: optuna.Trial,
    model_name: str,
    train_plans: list,
    val_plans: list,
    action_vocab: dict,
    goal_vocab: dict,
    max_plan_dim: int,
    epochs: int,
    batch_size: int,
    recurrent_type: str = "lstm",
    loss_function: str = "binary_crossentropy",
) -> float:
    """Optuna objective: train one trial and return the weighted average
    negative Hamming loss on the validation set."""

    if train_plans is None or action_vocab is None or goal_vocab is None:
        raise ValueError(
            ERRORS.MSG_ERROR_LOAD_PARAMS,
            f": train_plans={train_plans}, action_vocab={action_vocab}, "
            f"goal_vocab={goal_vocab}",
        )

    n_goals = len(goal_vocab)
    x_train, y_train = _prepare_xy(train_plans, n_goals)
    x_val, y_val = _prepare_xy(val_plans, n_goals)

    use_dropout = trial.suggest_categorical("use_dropout", [True, False])
    dropout = trial.suggest_float("dropout", 0, 0.5) if use_dropout else 0.0

    use_recurrent_dropout = trial.suggest_categorical("use_recurrent_dropout", [True, False])
    recurrent_dropout = (
        trial.suggest_float("recurrent_dropout", 0, 0.5)
        if use_recurrent_dropout
        else 0.0
    )

    params = ParamsGenerator(
        model_name=model_name,
        recurrent_type=trial.suggest_categorical("recurrent_type", [recurrent_type]),
        units=trial.suggest_int("hidden_layer_dim", 150, 512),
        output_dim=trial.suggest_int("embedding_dim", 50, 200),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        loss_function=trial.suggest_categorical("loss_function", [loss_function]),
    )
    params = params.generate(1)[0]
    print(params)

    model = build_network_single_fact(
        input_length=max_plan_dim,
        embedding_input_dim=len(action_vocab),
        output_size=n_goals,
        **params,
    )
    print_network_details(model, params)

    monitor = "val_neg_hamming_metric"
    callbacks = [
        EarlyStopping(**get_callback_default_params("early_stopping")),
        TFKerasPruningCallbackWarmup(
            trial, monitor, warmup_epochs=DEFAULTS.WARMUP_EPOCHS
        ),
    ]

    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    y_pred, y_true = get_model_predictions(model, x_val, y_val, batch_size)

    result = []
    n_goals = len(y_pred[0][0])
    for i in range(len(y_pred)):
        actual_pred = np.zeros(n_goals)
        actual_goal = np.zeros(n_goals)
        for j in range(len(y_pred[i])):
            actual_pred += y_pred[i][j]
            actual_goal += y_true[i][j]
            current_pred = [0 if pred < 0.5 else 1 for pred in actual_pred]
            current_goal = [0 if pred == 0 else 1 for pred in actual_goal]
        result.append(-sk_metrics.hamming_loss(current_goal, current_pred))

    weights = range(1, len(result) + 1)
    weighted_avg = sum(i * j for i, j in zip(result, weights)) / sum(weights)
    return weighted_avg


def run_tests(
    model,
    test_plans: list,
    action_vocab: dict,
    goal_vocab: dict,
    batch_size: int,
    max_plan_dim: int,
    min_plan_perc: float,
    plan_percentage: float,
    save_dir: str,
    filename: str = "metrics",
) -> None:
    """Run the model on a test set and print/save evaluation metrics.

    Args:
        model: Trained Keras model.
        test_plans: List of ``[action_ids, goal_index_sequence]`` pairs.
        action_vocab: Action name → integer ID mapping.
        goal_vocab: Predicate string → one-hot vector mapping.
        batch_size: Batch size for ``model.predict``.
        max_plan_dim: Fixed sequence length the model was built with.
        min_plan_perc: Minimum plan completion percentage (passed through to metrics).
        plan_percentage: Maximum plan completion percentage used during data loading.
        save_dir: Directory where the metrics file is written.
        filename: Base name for the output metrics file (no extension).
    """
    if test_plans is None:
        raise ValueError(ERRORS.MSG_ERROR_LOAD_PARAMS, f": test_plans={test_plans}")

    n_goals = len(goal_vocab)
    x_test, y_test = _prepare_xy(test_plans, n_goals)

    # If y_test is 2D (n_samples, n_goals) repeat across the time axis
    if y_test.ndim == 2:
        y_test = np.repeat(y_test[:, np.newaxis, :], x_test.shape[1], axis=1)

    y_pred, y_true = get_model_predictions(model, x_test, y_test, batch_size)
    print_metrics(
        y_true=y_true,
        y_pred=y_pred,
        goal_vocab=goal_vocab,
        save_dir=save_dir,
        filename=filename,
    )


def create_study(study_name: str, db_dir: str) -> optuna.Study:
    """Create or resume an Optuna study backed by a SQLite database.

    Args:
        study_name: Unique name for the study (also used as the DB filename stem).
        db_dir: Directory where the SQLite file is stored.

    Returns:
        An ``optuna.Study`` configured to maximise the objective.
    """
    return optuna.create_study(
        storage=f"sqlite:///{join(db_dir, f'{study_name}.db')}",
        sampler=TPESampler(seed=RANDOM_SEED),
        direction="maximize",
        load_if_exists=True,
        study_name=study_name,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.group()
def run():
    """CLERNet training CLI — train models and run Optuna hyperparameter search."""
    pass


@run.group("train-model")
@click.pass_context
@click.option(
    "--target-dir",
    "target_dir",
    required=True,
    prompt=True,
    type=click.STRING,
    help=f"{HELPS.MODEL_DIR_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
@click.option(
    "--plan-perc",
    "max_plan_percentage",
    required=True,
    prompt=True,
    help=HELPS.MAX_PLAN_PERCENTAGE,
    type=click.FloatRange(0, 1),
)
@click.option(
    "--batch-size",
    "batch_size",
    default=DEFAULTS.BATCH_SIZE,
    type=click.INT,
    help=HELPS.BATCH_SIZE,
    show_default=True,
)
@click.option(
    "--read-dict-dir",
    "read_dict_dir",
    required=True,
    prompt=True,
    help=HELPS.DICT_FOLDER_SRC,
    type=click.STRING,
)
@click.option(
    "--epochs",
    default=DEFAULTS.EPOCHS,
    help=HELPS.EPOCHS,
    type=click.INT,
    show_default=True,
)
@click.option(
    "--loss-function",
    default=DEFAULTS.LOSS_FUNCTION,
    help=HELPS.LOSS_FUNCTION,
    type=click.Choice(list(LOSSES_SHORT_NAMES.keys())),
    show_default=True,
)
@click.option(
    "--min-plan-perc",
    "min_plan_percentage",
    default=DEFAULTS.MIN_PLAN_PERC,
    type=click.FloatRange(0, 1),
    help=HELPS.MIN_PLAN_PERCENTAGE,
    show_default=True,
)
@click.option(
    "--read-plans-dir",
    "read_plans_dir",
    required=True,
    prompt=True,
    help=HELPS.PLANS_FOLDER_SRC,
    type=click.STRING,
)
@click.option(
    "--max-plan-dim",
    "max_plan_dim",
    required=True,
    prompt=True,
    help=HELPS.MAX_PLAN_LENGTH,
    type=click.INT,
)
def train_model(
    ctx,
    target_dir,
    max_plan_percentage,
    batch_size,
    read_dict_dir,
    epochs,
    loss_function,
    min_plan_percentage,
    read_plans_dir,
    max_plan_dim,
):
    """Load vocabularies and shared training options; propagate them via Click context."""
    print("Loading dictionaries...")
    [action_vocab, goal_vocab] = load_from_pickles(
        read_dict_dir, [FILENAMES.ACTION_DICT_FILENAME, FILENAMES.GOALS_DICT_FILENAME]
    )
    max_plan_dim = int(max_plan_percentage * max_plan_dim)

    if action_vocab is not None and goal_vocab is not None and max_plan_dim > 0:
        ctx.ensure_object(dict)
        ctx.obj[KEYS.READ_DICT_DIR] = read_dict_dir
        ctx.obj[KEYS.ACTION_DICT] = action_vocab
        ctx.obj[KEYS.GOALS_DICT] = goal_vocab
        ctx.obj[KEYS.EPOCHS] = epochs
        ctx.obj[KEYS.LOSS_FUNCTION] = loss_function
        ctx.obj[KEYS.BATCH_SIZE] = batch_size
        ctx.obj[KEYS.MAX_PLAN_PERC] = max_plan_percentage
        ctx.obj[KEYS.MIN_PLAN_PERC] = min_plan_percentage
        ctx.obj[KEYS.READ_PLANS_DIR] = read_plans_dir
        ctx.obj[KEYS.MAX_PLAN_DIM] = max_plan_dim
        ctx.obj[KEYS.TARGET_DIR] = target_dir


@train_model.group("neural-network")
@click.pass_context
@click.option(
    "--network-params",
    "params_dir",
    required=True,
    prompt=True,
    help=HELPS.NETWORK_PARAMETERS_SRC,
    type=click.STRING,
)
def neural_network(ctx, params_dir):
    """Load network hyperparameters from a JSON file and set up the model output directory."""
    if params_dir is not None and ctx.ensure_object(dict):
        target_dir = ctx.obj[KEYS.TARGET_DIR]
        epochs = ctx.obj[KEYS.EPOCHS]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]

        params = load_file(
            params_dir,
            load_ok=ERRORS.STD_LOAD_FILE_OK.format(
                os.path.basename(params_dir), os.path.dirname(params_dir)
            ),
            error=ERRORS.STD_ERROR_LOAD_FILE.format(os.path.basename(params_dir)),
        )
        model_dir_name = create_model_dir_name(params, epochs, max_plan_percentage, batch_size)
        model_dir = join(target_dir, model_dir_name)
        os.makedirs(model_dir, exist_ok=True)

        ctx.obj[KEYS.PARAMS] = params
        ctx.obj[KEYS.MODEL_NAME] = params["model_name"]
        ctx.obj[KEYS.MODEL_DIR] = model_dir


@neural_network.command("train")
@click.pass_context
@click.option(
    "--max-train-samples",
    "max_train_samples",
    default=None,
    type=click.INT,
    help="Cap the number of training samples (useful for quick smoke tests).",
    show_default=True,
)
@click.option(
    "--max-val-samples",
    "max_val_samples",
    default=None,
    type=click.INT,
    help="Cap the number of validation samples.",
    show_default=True,
)
def network_train(ctx, max_train_samples, max_val_samples):
    """Train a single model run with the loaded hyperparameters and save the result."""
    if ctx.ensure_object(dict):
        params = ctx.obj[KEYS.PARAMS]
        model_dir = ctx.obj[KEYS.MODEL_DIR]
        action_vocab = ctx.obj[KEYS.ACTION_DICT]
        goal_vocab = ctx.obj[KEYS.GOALS_DICT]
        epochs = ctx.obj[KEYS.EPOCHS]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]
        min_plan_percentage = ctx.obj[KEYS.MIN_PLAN_PERC]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]
        model_name = ctx.obj[KEYS.MODEL_NAME]

        plot_dir = join(model_dir, FILENAMES.NETWORK_PLOTS_FOLDER)
        os.makedirs(plot_dir, exist_ok=True)

        print("Loading training data...")
        [train_plans, val_plans] = load_from_pickles(
            read_plans_dir,
            [
                FILENAMES.COMPACT_TRAIN_PLANS_FILENAME_SIMPLE_JSON,
                FILENAMES.COMPACT_VALIDATION_PLANS_FILENAME_SIMPLE_JSON,
            ],
        )

        if train_plans is None or action_vocab is None or goal_vocab is None:
            raise ValueError(
                ERRORS.MSG_ERROR_LOAD_PARAMS,
                f": train_plans={train_plans}, action_vocab={action_vocab}, "
                f"goal_vocab={goal_vocab}",
            )

        np.random.seed(RANDOM_SEED)
        np.random.shuffle(train_plans)
        np.random.shuffle(val_plans)

        # Apply sample caps (defaults: 55 000 train / 5 000 val)
        train_cap = max_train_samples if max_train_samples is not None else 55000
        val_cap = max_val_samples if max_val_samples is not None else 5000
        train_plans = train_plans[:train_cap]
        val_plans = val_plans[:val_cap]

        print(f"Training on {len(train_plans)} samples, validating on {len(val_plans)}.")

        n_goals = len(goal_vocab)
        x_train, y_train = _prepare_xy(train_plans, n_goals)
        x_val, y_val = _prepare_xy(val_plans, n_goals)

        print(f"x_train[0][:5]={x_train[0][:5]}, y_train shape={y_train.shape}")

        model = build_network_single_fact(
            input_length=max_plan_dim,
            embedding_input_dim=len(action_vocab),
            output_size=n_goals,
            **params,
        )
        print_network_details(model, params)

        callbacks = [
            EarlyStopping(**get_callback_default_params("early_stopping")),
            ModelCheckpoint(
                filepath=join(model_dir, "checkpoint.keras"),
                **get_callback_default_params("model_checkpoint"),
            ),
        ]

        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
        )

        model.save(join(model_dir, f"{model_name}.keras"))
        json.dump(
            history.history,
            open(join(model_dir, f"{model_name}_training_history.json"), "w"),
        )
        save_all_history_plots(history=history, model_name=model_name, plot_dir=plot_dir)


@neural_network.command("results")
@click.pass_context
@click.option(
    "--incremental-tests",
    "incremental_tests",
    is_flag=True,
    default=False,
    help=HELPS.INCREMENTAL_TESTS_FLAG,
)
def network_results(ctx, incremental_tests):
    """Evaluate a trained model on the test set and print per-instance metrics."""
    if ctx.ensure_object(dict):
        params = ctx.obj[KEYS.PARAMS]
        model_dir = ctx.obj[KEYS.MODEL_DIR]
        action_vocab = ctx.obj[KEYS.ACTION_DICT]
        goal_vocab = ctx.obj[KEYS.GOALS_DICT]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]
        min_plan_percentage = ctx.obj[KEYS.MIN_PLAN_PERC]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]

        model_name = params["model_name"]
        model = load_model(
            join(model_dir, f"{model_name}.keras"),
            custom_objects=CUSTOM_OBJECTS,
        )
        print(model.summary())

        if incremental_tests:
            test_dir = join(read_plans_dir, "incremental_test_sets")
            files = os.listdir(test_dir)
        else:
            test_dir = read_plans_dir
            files = [FILENAMES.TEST_PLANS_FILENAME_SIMPLE_JSON]

        for f in files:
            test_plans = load_file(join(test_dir, f))
            if test_plans is not None and len(test_plans) > 0:
                run_tests(
                    model=model,
                    test_plans=test_plans,
                    action_vocab=action_vocab,
                    goal_vocab=goal_vocab,
                    batch_size=batch_size,
                    max_plan_dim=max_plan_dim,
                    min_plan_perc=min_plan_percentage,
                    plan_percentage=max_plan_percentage,
                    save_dir=model_dir,
                    filename=f"{model_name}_metrics_{f.split('.')[0]}",
                )
            else:
                print(f"Problems with file {f} in folder {test_dir}")


@neural_network.command("goal-rec-results")
@click.pass_context
@click.option(
    "--read-test-plans-dir",
    "read_test_plans_dir",
    required=True,
    prompt=True,
    help=HELPS.GOAL_REC_TEST_PLANS_DIR,
)
@click.option(
    "--fast",
    "fast",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Vectorize operations where possible.",
)
def network_goal_rec_results(ctx, read_test_plans_dir, fast):
    """Run goal recognition prediction on test instances via an external script."""
    if ctx.ensure_object(dict):
        model_path = f"{ctx.obj[KEYS.MODEL_DIR]}/{ctx.obj[KEYS.MODEL_NAME]}.keras"
        read_dict_dir = ctx.obj[KEYS.READ_DICT_DIR]
        target_dir = f"{ctx.obj[KEYS.MODEL_DIR]}/goal_rec_results/"
        max_dim = ctx.obj[KEYS.MAX_PLAN_DIM]
        max_plan_percentage = ctx.obj[KEYS.MAX_PLAN_PERC]

        ctx.obj[KEYS.GOAL_REC_RESULTS_DIR] = target_dir

        cmd = [
            "python", "get_predictions_and_results.py",
            "--model-path", model_path,
            "--read-dict-dir", read_dict_dir,
            "--read-test-plans-dir", read_test_plans_dir,
            "--target-dir", target_dir,
            "--max-plan-dim", str(max_dim),
            "--max-plan-perc", str(max_plan_percentage),
        ]
        if fast:
            cmd.append("--fast")
        subprocess.run(cmd)


@neural_network.command("goal-rec-result-metrics")
@click.pass_context
@click.option(
    "--domain-name",
    "domain_name",
    required=True,
    prompt=True,
    help=HELPS.GOAL_REC_TEST_PLANS_DIR,
)
def network_goal_rec_result_metrics(ctx, domain_name):
    """Aggregate goal recognition result files into summary metrics via an external script."""
    if ctx.ensure_object(dict):
        goal_rec_results_dir = f"{ctx.obj[KEYS.MODEL_DIR]}/goal_rec_results/"
        model_name = ctx.obj[KEYS.MODEL_NAME]
        target_dir = f"{ctx.obj[KEYS.MODEL_DIR]}/"

        subprocess.run([
            "python", "compute_goal_rec_result_metrics.py",
            "--domain-name", domain_name,
            "--model-name", model_name,
            "--goal-rec-results-dir", goal_rec_results_dir,
            "--target-dir", target_dir,
        ])


@train_model.command("optuna")
@click.pass_context
@click.option(
    "--model-name",
    "model_name",
    type=click.STRING,
    required=True,
    prompt=True,
    help=HELPS.MODEL_NAME,
)
@click.option(
    "--trials",
    "n_trials",
    default=DEFAULTS.OPTUNA_TRIALS,
    type=click.INT,
    help=HELPS.TRIALS,
    show_default=True,
)
@click.option(
    "--db-dir",
    "db_dir",
    type=click.STRING,
    required=True,
    prompt=True,
    help=HELPS.DB_DIR,
)
@click.option(
    "--max-train-samples",
    "max_train_samples",
    default=None,
    type=click.INT,
    help="Cap the number of training samples per trial.",
)
@click.option(
    "--max-val-samples",
    "max_val_samples",
    default=None,
    type=click.INT,
    help="Cap the number of validation samples per trial.",
)
def optuna_train(ctx, model_name, db_dir, n_trials, max_train_samples, max_val_samples):
    """Run an Optuna hyperparameter search and save visualizations when available."""
    if ctx.ensure_object(dict):
        action_vocab = ctx.obj[KEYS.ACTION_DICT]
        goal_vocab = ctx.obj[KEYS.GOALS_DICT]
        batch_size = ctx.obj[KEYS.BATCH_SIZE]
        read_plans_dir = ctx.obj[KEYS.READ_PLANS_DIR]
        max_plan_dim = ctx.obj[KEYS.MAX_PLAN_DIM]
        epochs = ctx.obj[KEYS.EPOCHS]
        target_dir = ctx.obj[KEYS.TARGET_DIR]
        loss_function = ctx.obj[KEYS.LOSS_FUNCTION]

        os.makedirs(db_dir, exist_ok=True)
        study = create_study(model_name, db_dir)

        ctx.obj[KEYS.STUDY] = study
        ctx.obj[KEYS.MODEL_NAME] = model_name

        print("Loading training data for Optuna...")
        [train_plans, val_plans] = load_from_pickles(
            read_plans_dir,
            [
                FILENAMES.COMPACT_TRAIN_PLANS_FILENAME_SIMPLE_JSON,
                FILENAMES.COMPACT_VALIDATION_PLANS_FILENAME_SIMPLE_JSON,
            ],
        )

        np.random.shuffle(train_plans)
        np.random.shuffle(val_plans)

        train_cap = max_train_samples if max_train_samples is not None else 55000
        val_cap = max_val_samples if max_val_samples is not None else 5000
        train_plans = train_plans[:train_cap]
        val_plans = val_plans[:val_cap]

        study.optimize(
            lambda trial: objective(
                trial=trial,
                model_name=model_name,
                train_plans=train_plans,
                val_plans=val_plans,
                action_vocab=action_vocab,
                goal_vocab=goal_vocab,
                max_plan_dim=max_plan_dim,
                epochs=epochs,
                batch_size=batch_size,
                loss_function=loss_function,
            ),
            n_trials=n_trials,
            gc_after_trial=True,
        )

        plot_dir = join(target_dir, model_name, FILENAMES.NETWORK_PLOTS_FOLDER)
        os.makedirs(plot_dir, exist_ok=True)

        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(join(plot_dir, "opt_hist.png"))

            fig = optuna.visualization.plot_slice(study)
            fig.write_image(join(plot_dir, "slice.png"))

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(join(plot_dir, "param_importance.png"))
        except ImportError:
            print("[WARN] plotly not installed — skipping Optuna visualizations")


@run.command("optuna-results")
@click.option("--model-name", "model_name", type=click.STRING, required=True, prompt=True, help=HELPS.MODEL_NAME)
@click.option("--db-dir", "db_dir", type=click.STRING, required=True, prompt=True, help=HELPS.DB_DIR)
def optuna_results(model_name, db_dir):
    """Print the best hyperparameters found in a completed Optuna study."""
    study = create_study(model_name, db_dir)
    print(study.best_params)


@run.command("optuna-create-best-params-json")
@click.option("--model-name", "model_name", type=click.STRING, required=True, prompt=True, help=HELPS.MODEL_NAME)
@click.option("--db-dir", "db_dir", type=click.STRING, required=True, prompt=True, help=HELPS.DB_DIR)
@click.option(
    "--target-dir",
    "target_dir",
    default=os.getcwd(),
    type=click.STRING,
    help=f"{HELPS.MODEL_DIR_OUT} {HELPS.CREATE_IF_NOT_EXISTS}",
)
def optuna_create_best_params_json(model_name, db_dir, target_dir):
    """Export the best Optuna trial's hyperparameters to a JSON file for use with clernet-train."""
    study = create_study(model_name, db_dir)
    print(study.best_params)
    best_params = study.best_params

    # Rename keys to match ParamsGenerator expectations
    best_params["output_dim"] = best_params.pop("embedding_dim")
    best_params["units"] = best_params.pop("hidden_layer_dim")

    params_gen = ParamsGenerator(model_name=model_name, **best_params)
    params = params_gen.generate(1)[0]

    json_params_filename = model_name + "_params.json"
    with open(join(target_dir, json_params_filename), "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    RANDOM_SEED = 1000
    np.random.seed(RANDOM_SEED)
    run()
