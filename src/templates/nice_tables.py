import pandas as pd
import numpy as np
from decimal import Decimal


def alpha2_table(table):
    tmp_alpha2 = table.copy()
    one_alpha = table.alpha.unique()[0]
    one_penalty = table.penalty.unique()[0]
    one_optimizer = table.optimizer.unique()[0]
    one_lambda = table["lambda"].unique()[0]
    tmp_alpha2 = tmp_alpha2[
        (tmp_alpha2.alpha == one_alpha)
        & (tmp_alpha2.penalty == one_penalty)
        & (tmp_alpha2.optimizer == one_optimizer)
        & (tmp_alpha2["lambda"] == one_lambda)
    ]
    tmp_alpha2 = tmp_alpha2.reset_index()
    tmp_alpha2["data_name"] = tmp_alpha2.apply(
        lambda row: row["run"] + " (" + str(row["n"]) + "," + str(row["p"]) + ")",
        axis=1,
    )

    def custom_agg_func(array):
        mean = np.mean(array)
        std = np.std(array)
        res = f"{Decimal(mean):.2E} +/- {Decimal(std):.2E}"
        return res

    new_table = pd.pivot_table(
        tmp_alpha2,
        values="alpha2",
        index="data_name",
        columns="name",
        aggfunc=custom_agg_func,
    )
    return new_table
