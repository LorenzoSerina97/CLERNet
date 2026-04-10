import os

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics as sk_metrics
from keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    GRU,
    Input,
    LSTM,
    TimeDistributed,
)
from keras.losses import Loss
from keras.metrics import Precision, Recall
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2, l2
from os.path import join
from typing import Union

from src.constants import ERRORS
from src.models.attention import AttentionWeights, ContextVector
from src.models.loss import (
    bce_hmlp,
    bce_mlp,
    bce_ol,
    bce_op,
    bfce_hmlp,
    bfce_mlp,
    bfce_ol,
    bfce_op,
    neg_hamming_metric,
    rmse,
    rmse_hmlp,
    rmse_mlp,
    rmse_ol,
    rmse_op,
)
from src.utils import create_table

LOSS_FUNCTIONS_MAP = {
    "rmse": rmse,
    "rmse_ol": rmse_ol,
    "rmse_op": rmse_op,
    "rmse_hmlp": rmse_hmlp,
    "rmse_mlp": rmse_mlp,
    "bce_ol": bce_ol,
    "bce_op": bce_op,
    "bce_hmlp": bce_hmlp,
    "bce_mlp": bce_mlp,
    "bfce_ol": bfce_ol,
    "bfce_op": bfce_op,
    "bfce_hmlp": bfce_hmlp,
    "bfce_mlp": bfce_mlp,
}

# All custom objects needed to reload a saved model.
CUSTOM_OBJECTS = {
    "AttentionWeights": AttentionWeights,
    "ContextVector": ContextVector,
    "neg_hamming_metric": neg_hamming_metric,
    **LOSS_FUNCTIONS_MAP,
}


def build_network_single_fact(
    input_length: int,
    embedding_input_dim: int,
    output_size: int,
    embedding_params: dict = None,
    hidden_layers: int = 1,
    regularizer_params: dict = None,
    recurrent_list: list = ["lstm", None],
    use_attention: bool = False,
    use_time_distributed: bool = True,
    optimizer_list: list = ["adam", None],
    loss_function: Union[Loss, str] = "binary_crossentropy",
    model_name: str = "model",
) -> Model:
    inputs = Input(shape=(input_length,))
    prev_layer = inputs

    if embedding_params is not None:
        embedding_layer = Embedding(
            input_dim=embedding_input_dim + 1,
            input_length=input_length,
            **embedding_params,
        )(prev_layer)
        prev_layer = embedding_layer

    for layer in range(hidden_layers):
        if regularizer_params is None or (
            regularizer_params["l1"] is None and regularizer_params["l2"] is None
        ):
            regularizer = None
        elif regularizer_params["l1"] is None:
            regularizer = l2(regularizer_params["l2"])
        elif regularizer_params["l2"] is None:
            regularizer = l1(regularizer_params["l1"])
        else:
            regularizer = l1_l2(
                l1=regularizer_params["l1"], l2=regularizer_params["l2"]
            )

        recurrent_type, recurrent_params = recurrent_list
        if recurrent_type == "lstm":
            recurrent_layer = LSTM(
                **recurrent_params,
                name=f"lstm_layer_{layer}",
            )(prev_layer)
        elif recurrent_type == "gru":
            recurrent_layer = GRU(
                **recurrent_params,
                name=f"gru_layer_{layer}",
            )(prev_layer)
        elif recurrent_type == "bilstm":
            lstm = LSTM(**recurrent_params)
            recurrent_layer = Bidirectional(
                layer=lstm,
                name=f"bilstm_layer_{layer}",
            )(prev_layer)
        prev_layer = recurrent_layer

    if use_time_distributed and use_attention:
        print(ERRORS.MSG_ERROR_BOTH_ATTENTION_TIME_DISTRIB)
        return None

    if use_attention:
        attention_weights = AttentionWeights(input_length, name="attention_weights")(
            prev_layer
        )
        context_vector = ContextVector()([prev_layer, attention_weights])
        prev_layer = context_vector

    if use_time_distributed:
        output_dense = Dense(output_size, activation="sigmoid")
        outputs = TimeDistributed(output_dense)(prev_layer)
    else:
        outputs = Dense(output_size, activation="sigmoid", name="output")(prev_layer)

    optimizer_type, optimizer_params = optimizer_list
    if optimizer_type == "adam":
        optimizer = Adam(**optimizer_params)

    # Resolve loss function name to callable if needed
    resolved_loss = LOSS_FUNCTIONS_MAP.get(loss_function, loss_function)

    training_metrics = [
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall"),
        neg_hamming_metric,
    ]

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=optimizer,
        loss=resolved_loss,
        metrics=training_metrics,
    )

    return model


def print_network_details(model: Model, params: dict, save_file: str = None) -> None:
    headers = [
        "EMBEDDING DIM",
        "LOSS FUNCTION",
        "RECURRENT DIM",
        "DROPOUT",
        "RECURRENT_DROPOUT",
    ]
    rows = [
        [
            params["embedding_params"]["output_dim"],
            params["loss_function"],
            params["recurrent_list"][1]["units"],
            params["recurrent_list"][1]["dropout"],
            params["recurrent_list"][1]["recurrent_dropout"],
        ]
    ]
    title = f"{model.name} details"
    to_print = create_table(title, headers, rows, just=18)
    to_print.append(model.summary())
    if save_file is None:
        for line in to_print:
            print(line)
    else:
        with open(save_file, "w") as f:
            for line in to_print:
                f.write(line)


def save_all_history_plots(
    history: dict, model_name: str = "Model", plot_dir: str = None
) -> None:
    if plot_dir is None:
        plot_dir = os.getcwd()

    plottable_metrics = [
        metric
        for metric in history.history.keys()
        if ("val_" not in metric) and ("val_" + metric in history.history.keys())
    ]

    for metric in plottable_metrics:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title(f"{model_name} {metric.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{metric}_plot.png"), bbox_inches="tight")
        plt.close()


def print_metrics(
    y_true: list,
    y_pred: list,
    goal_vocab: dict,
    save_dir: str = None,
    filename: str = "metrics",
) -> list:
    """Compute and print/save evaluation metrics for the model.

    Args:
        y_true: Ground-truth label sequences (list of per-step label arrays).
        y_pred: Predicted label sequences (list of per-step prediction arrays).
        goal_vocab: Dictionary mapping goal strings to one-hot index vectors.
        save_dir: Directory to save the metrics file; prints to stdout if None.
        filename: Base name for the output metrics file.

    Returns:
        [accuracy, hamming_loss] averaged over all samples.
    """
    result = []
    accuracy_tot = []
    f1_tot = []
    precision_tot = []
    recall_tot = []

    for i in range(len(y_pred)):
        result_track = []
        accuracy_track = []
        f1_track = []
        precision_track = []
        recall_track = []
        actual_pred = np.zeros(y_pred[0][0].shape)
        actual_goal = np.zeros(y_pred[0][0].shape)
        for j in range(len(y_pred[i])):
            actual_pred += y_pred[i][j]
            actual_goal += y_true[i][j]
            current_pred = [0 if pred < 0.5 else 1 for pred in actual_pred]
            current_goal = [0 if pred == 0 else 1 for pred in actual_goal]

            accuracy_track.append(sk_metrics.accuracy_score(current_goal, current_pred))
            result_track.append(-sk_metrics.hamming_loss(current_goal, current_pred))
            f1_track.append(sk_metrics.f1_score(current_goal, current_pred, zero_division=0))
            precision_track.append(
                sk_metrics.precision_score(current_goal, current_pred, zero_division=0)
            )
            recall_track.append(
                sk_metrics.recall_score(current_goal, current_pred, zero_division=0)
            )

        weights = range(1, len(result_track) + 1)
        hamming_loss = sum(i * j for i, j in zip(result_track, weights)) / sum(weights)
        weights_acc = range(1, len(accuracy_track) + 1)
        accuracy = sum(i * j for i, j in zip(accuracy_track, weights_acc)) / sum(
            weights_acc
        )
        weights_f1 = range(1, len(f1_track) + 1)
        f1 = sum(i * j for i, j in zip(f1_track, weights_f1)) / sum(weights_f1)
        weights_precision = range(1, len(precision_track) + 1)
        precision = sum(i * j for i, j in zip(precision_track, weights_precision)) / sum(
            weights_precision
        )
        weights_recall = range(1, len(recall_track) + 1)
        recall = sum(i * j for i, j in zip(recall_track, weights_recall)) / sum(
            weights_recall
        )
        f1_tot.append(f1)
        precision_tot.append(precision)
        recall_tot.append(recall)
        result.append(hamming_loss)
        accuracy_tot.append(accuracy)

    hamming_loss = sum(result) / len(result)
    accuracy = sum(accuracy_tot) / len(accuracy_tot)

    to_print = []
    to_print.append(f"Accuracy: {accuracy}\n")
    to_print.append(f"Hamming Loss: {hamming_loss}\n")
    to_print.append(f"F1: {sum(f1_tot) / len(f1_tot)}\n")
    to_print.append(f"Precision: {sum(precision_tot) / len(precision_tot)}\n")
    to_print.append(f"Recall: {sum(recall_tot) / len(recall_tot)}\n")

    if save_dir is None:
        for line in to_print:
            print(line)
    else:
        with open(join(save_dir, f"{filename}.txt"), "w") as file:
            for line in to_print:
                file.write(line)
    return [accuracy, hamming_loss]


def get_model_predictions(model: Model, x_test, y_test, batch_size) -> list:
    y_pred = model.predict(x_test, batch_size=batch_size)
    return y_pred.tolist(), y_test.tolist()


def create_model_dir_name(
    params: dict, epochs: int, max_plan_percentage: float, batch_size: int
) -> str:
    hidden_layer_dim = params["recurrent_list"][1]["units"]
    embedding_dim = params["embedding_params"]["output_dim"]
    dropout = params["recurrent_list"][1]["dropout"]
    recurrent_dropout = params["recurrent_list"][1]["recurrent_dropout"]
    model_name = params["model_name"]
    recurrent_type = params["recurrent_list"][0]
    loss_function = params["loss_function"]

    return (
        f"{model_name}_{recurrent_type}_epochs={epochs}_embedding={embedding_dim}_units="
        f"{hidden_layer_dim}_dropout={dropout}_recurrent-dropout={recurrent_dropout}_loss="
        f"{loss_function}_plan-percentage={max_plan_percentage}_batch-size={batch_size}"
    )


def get_callback_default_params(callback_name: str) -> dict:
    if callback_name == "model_checkpoint":
        return {
            "save_weights_only": False,
            "monitor": "val_loss",
            "mode": "auto",
            "save_best_only": True,
        }
    if callback_name == "early_stopping":
        return {
            "monitor": "val_loss",
            "verbose": 2,
            "patience": 5,
            "mode": "min",
            "restore_best_weights": True,
        }
    return {}
