import json
import os
import pickle
import warnings
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def load_file(
    read_file: str,
    load_ok: str = "File loaded",
    error: str = f"Error while loading file",
) -> object:
    try:
        if read_file.endswith(".json") or read_file.endswith(".JSON"):
            with open(read_file, "r") as rf:
                o = json.load(rf)
        else:
            with open(read_file, "rb") as rf:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    o = pickle.load(rf)
        print(load_ok)
    except FileNotFoundError:
        print(error)
        o = None

    return o


def load_from_pickles(read_dir: str, files: list) -> list:
    MSG_OK = "{0} loaded from {1}"
    MSG_ERROR = "Could not load {0} from {1}.\nPlease check the -r parameter is correct"

    to_return = []
    for file_name in files:
        to_return.append(
            load_file(
                join(read_dir, file_name),
                load_ok=MSG_OK.format(file_name, read_dir),
                error=MSG_ERROR.format(file_name, read_dir),
            )
        )
    return to_return


def save_file(o: object, target_dir: str, filename: str) -> None:
    MSG_OK = "{0} saved in {1}"
    MSG_ERROR = "Could not save {0} in {1}"

    os.makedirs(target_dir, exist_ok=True)
    try:
        with open(join(target_dir, filename), "wb") as wf:
            pickle.dump(o, wf)
            wf.close()
        print(MSG_OK.format(filename, target_dir))
    except pickle.PicklingError:
        print(MSG_ERROR.format(filename, target_dir))

    return


def create_table(
    title: str, headers: list, rows: list, just: int = 10, precision: int = 2
) -> list:
    to_ret = []
    divider = "-" * ((just + 5) * len(headers) - 4)
    to_ret.append(title)
    to_ret.append(divider)
    s = ""
    for h in headers:
        s += f"{str(h).rjust(just)}     "
    to_ret.append(s)
    to_ret.append(divider)

    for row in rows:
        s = ""
        for value in row:
            if type(value) == int:
                s += f"{value:{just}.{precision}f}     "
            elif type(value) == float or type(value) == np.float64:
                s += f"{value:{just}.{precision}f}     "
            elif type(value) == str:
                s += f"{value.rjust(just)}     "
            else:
                print(f"Could not parse type {type(value)}")
        to_ret.append(s)
    to_ret.append(divider)
    return to_ret


def create_plot(plot_type: str, target_dir: str = None, **plt_kwargs) -> None:
    fig, ax = plt.subplots()
    if plot_type == "hist":
        create_histogram(ax, **plt_kwargs)
    elif plot_type == "simple":
        create_simpleplot(ax, **plt_kwargs)
    else:
        return

    fig.savefig(target_dir, bbox_inches="tight")
    plt.close(fig)

    # if target_dir is not None:
    #     fig.savefig(target_dir, bbox_inches="tight")
    #     plt.show(fig)
    #     plt.close()
    # else:
    #     plt.show(fig)
    #     plt.close()


def create_histogram(ax: Axes, input: list, nbins: int) -> None:
    min_freq = min(input)
    max_freq = max(input)
    bins = np.linspace(min_freq - 1, max_freq, nbins + 1)
    for i, el in enumerate(input):
        in_while = True
        bin_num = 0
        while in_while:
            if bins[bin_num] < el <= bins[bin_num + 1]:
                input[i] = np.mean([bins[bin_num], bins[bin_num + 1]])
                in_while = False
            else:
                bin_num += 1
    n, bins, patches = ax.hist(input, bins=bins, edgecolor="black")


def create_simpleplot(
    ax: Axes, x: list, y: list, xlabel: str = None, ylabel: str = None
) -> None:
    for label, data in y:
        if x is None:
            ax.plot(data, label=label)
        else:
            ax.plot(x, data, label=label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
