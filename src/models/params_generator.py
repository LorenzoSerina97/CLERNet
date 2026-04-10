from typing import Union, Any
import random
from keras.losses import Loss
import numpy as np
from pandas import DataFrame

from src.constants import RANDOM_SEED


def fun(l: list, elements: list, df: DataFrame):
    if len(elements) == 0:
        return [l]
    element = elements.pop(0)

    to_return = list()
    for i, el in enumerate(df.loc[element]["Values"]):
        k = l.copy()
        w = elements.copy()
        k.append(i)
        to_return.extend(fun(k, w, df))
    return to_return


class ParamsGenerator:
    k1 = "Name"
    k2 = "Values"

    def __init_table(self) -> None:
        self.table = DataFrame(columns=[self.k1, self.k2])
        self.table.loc[0] = ["output_dim", [124]]
        self.table.loc[1] = ["mask_zero", [True]]
        self.table.loc[2] = ["recurrent_type", None]
        self.table.loc[3] = ["units", [64]]
        self.table.loc[4] = ["activation", ["linear"]]
        self.table.loc[5] = ["return_sequences", [True]]
        self.table.loc[6] = ["dropout", [0.0]]
        self.table.loc[7] = ["recurrent_dropout", [0.0]]
        self.table.loc[8] = ["use_attention", [False]]
        self.table.loc[9] = ["optimizer_type", ["adam"]]
        self.table.loc[10] = ["learning_rate", [0.001]]
        self.table.loc[11] = ["beta_1", [0.9]]
        self.table.loc[12] = ["beta_2", [0.999]]
        self.table.loc[13] = ["amsgrad", [False]]
        self.table.loc[14] = ["clipnorm", [None]]
        self.table.loc[15] = ["loss_function", None]
        self.table.loc[16] = ["model_name", None]
        self.table.loc[17] = ["hidden_layers", [1]]
        self.table.loc[18] = ["l1", [None]]
        self.table.loc[19] = ["l2", [None]]
        self.table.loc[20] = ["use_time_distributed", [True]]

    @staticmethod
    def __create_list(element: Any, new_val: Any):
        if new_val is not None:
            current_element = new_val
        else:
            current_element = element

        if type(current_element) == set:
            return [el for el in list(current_element)]
        elif type(current_element) == list:
            if len(current_element) == 2:
                if type(element) == int:
                    step = 1
                elif type(element) == float:
                    step = (current_element[1] - current_element[0]) / 10
                else:
                    step = 0
                    print(type(element), element, current_element)
                return [
                    el
                    for el in list(
                        np.arange(current_element[0], current_element[1] + step, step)
                    )
                ]
        else:
            return [current_element]

    def __choice_to_dict(self, choice: list) -> dict:
        return {
            "embedding_params": {
                "output_dim": self.table.loc[0][self.k2][choice[0]],
                "mask_zero": self.table.loc[1][self.k2][choice[1]],
            },
            "hidden_layers": self.table.loc[17][self.k2][choice[17]],
            "regularizer_params": {
                "l1": self.table.loc[18][self.k2][choice[18]],
                "l2": self.table.loc[19][self.k2][choice[19]],  # row 19 = l2
            },
            "recurrent_list": [
                self.table.loc[2][self.k2][choice[2]],
                {
                    "units": self.table.loc[3][self.k2][choice[3]],
                    "activation": self.table.loc[4][self.k2][choice[4]],
                    "return_sequences": self.table.loc[5][self.k2][choice[5]],
                    "dropout": self.table.loc[6][self.k2][choice[6]],
                    "recurrent_dropout": self.table.loc[7][self.k2][choice[7]],
                },
            ],
            "use_attention": self.table.loc[8][self.k2][choice[8]],
            "use_time_distributed": self.table.loc[20][self.k2][choice[20]],
            "optimizer_list": [
                self.table.loc[9][self.k2][choice[9]],
                {
                    "learning_rate": self.table.loc[10][self.k2][choice[10]],
                    "beta_1": self.table.loc[11][self.k2][choice[11]],
                    "beta_2": self.table.loc[12][self.k2][choice[12]],
                    "amsgrad": self.table.loc[13][self.k2][choice[13]],
                    "clipnorm": self.table.loc[14][self.k2][choice[14]],
                },
            ],
            "loss_function": self.table.loc[15][self.k2][choice[15]],
            "model_name": self.table.loc[16][self.k2][choice[16]],
        
        }

    def __init__(
        self,
        model_name: str,
        recurrent_type: Union[str, set] = "lstm",
        loss_function: Union[str, set, Loss] = "binary_crossentropy",
        **kwargs
    ):
        self.__init_table()
        self.table.at[15, self.k2] = self.__create_list(
            self.table.at[15, self.k2], loss_function
        )
        self.table.at[16, self.k2] = self.__create_list(
            self.table.at[16, self.k2], model_name
        )
        self.table.at[2, self.k2] = self.__create_list(
            self.table.at[2, self.k2], recurrent_type
        )

        for i, row in self.table.iterrows():
            if row[self.k1] in kwargs.keys():
                self.table.at[i, self.k2] = self.__create_list(
                    self.table.at[i, self.k2][0], kwargs[row[self.k1]]
                )
        self.choices = fun([], list(self.table.index), self.table)
        random.shuffle(self.choices)

    def generate(self, num_instances: int) -> list:
        to_return = []
        for _ in range(min(num_instances, len(self.choices))):
            choice = self.choices.pop()
            to_return.append(self.__choice_to_dict(choice))

        return to_return


random.seed(RANDOM_SEED)
