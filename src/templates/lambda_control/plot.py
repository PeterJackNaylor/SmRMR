#!/usr/bin/env python
"""
Input variables:
    - FILE: path containing all simulated data
Output files:
    - *.png files
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from colors import (
    color_dictionnary_fdr,
    color_dictionnary_fdr_keys,
    name_mapping_fdr,
    mapping_data_name,
)

if not os.path.isdir("loss"):
    os.mkdir("loss")
if not os.path.isdir("alpha_fdr"):
    os.mkdir("alpha_fdr")
if not os.path.isdir("selected_features"):
    os.mkdir("selected_features")

row_dic = {100: 1, 500: 2, 5000: 3}
col_dic = {100: 1, 500: 2, 1000: 3}

titles = tuple(
    f"<b> {text} </b>" if text != "" else text
    for text in [
        "n = 100; p = 100",
        "n = 500; p = 100",
        "",
        "n = 100; p = 500",
        "n = 500; p = 500",
        "",
        "n = 100; p = 5000",
        "n = 500; p = 5000",
        "n = 1000; p = 5000",
    ]
)

tikz_y = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
tikz_text_y = ["-0.50", "-0.25", "0.00", "0.25", "0.50", "0.75", "1.00"]

tikz_x = [0.25, 0.5, 0.75]
tikz_text_x = ["0.25", "0.50", "0.75"]

id_ = [i / 100 for i in range(-100, 105, 5)]
flat_line = [0 for el in id_]


def count_selected(str_list):
    if "," not in str_list:
        return 0
    else:
        elements = str_list.split(",")
        return len(elements)


def new_name(am, ker):
    if am == "HSIC":
        return f"HSIC ({ker})"
    else:
        return am


def main():
    # Load data
    # table = pd.read_csv("${FILE}", sep="\t")
    table = pd.read_csv("performance.tsv", sep="\t")
    # set to 0 those which didn't select anything
    table.loc[table["value"] == -1, "value"] = 0
    table["kernel"] = table["kernel"].fillna("unspecified")
    table = table.dropna()
    table = table.loc[table["run"] != 0]
    table.loc[table["penalty"] == "none", "lamb"] = table["lamb"].min() / 10

    tmp_pen = [el for el in table["penalty"].unique() if el != "none"]
    tmp_tab = table[table["penalty"] == "none"]
    for pen in tmp_pen:
        tmp_tab.loc[:, "penalty"] = pen
        table = pd.concat([table, tmp_tab])
    table = table[table["penalty"] != "none"]
    datasets = np.unique(table["run"])
    table["name"] = table.apply(lambda row: new_name(row["AM"], row["kernel"]), axis=1)
    group_optim_penal = table.groupby(["optimizer", "penalty"])

    for (opti, penal), sub_table in group_optim_penal:
        for data in list(datasets):
            table_data = sub_table.loc[sub_table["run"] == data]

            groups = table_data.groupby(["n", "p", "AM", "kernel", "name"])
            fig = make_subplots(
                rows=3,
                cols=3,
                shared_xaxes=True,
                shared_yaxes=True,
                vertical_spacing=0.06,
                horizontal_spacing=0.04,
                subplot_titles=titles,
                x_title="lambda",  # , 'font': {'size': 0}},
                y_title="<i>&#945;</i> - <i>FDR</i>",
            )

            fig_loss = make_subplots(
                rows=3,
                cols=3,
                shared_xaxes=True,
                shared_yaxes=False,
                vertical_spacing=0.06,
                horizontal_spacing=0.04,
                subplot_titles=titles,
                x_title="lambda",  # , 'font': {'size': 0}},
                y_title="loss",
            )

            def makeplot():
                return make_subplots(
                    rows=3,
                    cols=3,
                    vertical_spacing=0.06,
                    horizontal_spacing=0.04,
                    subplot_titles=titles,
                    specs=[
                        [{"is_3d": True}, {"is_3d": True}, {"is_3d": True}],
                        [{"is_3d": True}, {"is_3d": True}, {"is_3d": True}],
                        [{"is_3d": True}, {"is_3d": True}, {"is_3d": True}],
                    ],
                    print_grid=False,
                )

            fig_am = {el: makeplot() for el in table_data.name.unique()}
            legend = {el: True for el in color_dictionnary_fdr_keys}
            legend_loss = legend.copy()
            for g_n, group in groups:

                n = int(g_n[0])
                p = int(g_n[1])
                name = g_n[2]
                kernel = g_n[3]
                group["control_diff"] = group["alpha"] - group["value"]
                lamb_group = group.groupby(["lamb"])
                mean = lamb_group.mean()
                sample_number = lamb_group.count()
                std = lamb_group.std()

                # plot alpha - fdr
                x = mean.index.sort_values()
                y = mean.loc[x, "control_diff"]
                std_y = std.loc[x, "control_diff"]
                n_samples = sample_number.loc[x, "control_diff"]
                err = 1.96 * std_y / (n_samples) ** 0.5
                curve_name = name_mapping_fdr(name, kernel)
                log_scale = np.log(x)
                curve = go.Scatter(
                    x=log_scale,
                    y=y,
                    name=curve_name,
                    error_y=dict(array=err),
                    marker=dict(
                        color=color_dictionnary_fdr(
                            name,
                            kernel,  # only_kernel=only_kernel
                        )
                    ),
                    showlegend=legend[curve_name],
                )

                if legend[curve_name]:
                    legend[curve_name] = False

                fig.add_trace(curve, row=row_dic[p], col=col_dic[n])
                fig.add_trace(
                    go.Scatter(
                        x=log_scale,
                        y=[0 for el in list(log_scale)],
                        name="",
                        marker={"color": "rgb(0, 0, 0)"},
                        line=dict(width=0.5),
                        showlegend=False,
                    ),
                    row=row_dic[p],
                    col=col_dic[n],
                )

                # plot loss vs lambda
                alpha_group = group[group["alpha"] == 0.1]
                lamb_group = alpha_group.groupby(["lamb"])
                mean = lamb_group.mean()
                sample_number = lamb_group.count()
                std = lamb_group.std()

                x = mean.index.sort_values()
                y = mean.loc[x, "loss"]
                std_y = std.loc[x, "loss"]
                n_samples = sample_number.loc[x, "loss"]
                err = 1.96 * std_y / (n_samples) ** 0.5
                curve_name = name_mapping_fdr(name, kernel)
                log_scale = np.log(x)
                curve = go.Scatter(
                    x=log_scale,
                    y=y,
                    name=curve_name,
                    error_y=dict(array=err),
                    marker=dict(
                        color=color_dictionnary_fdr(
                            name,
                            kernel,  # only_kernel=only_kernel
                        )
                    ),
                    showlegend=legend_loss[curve_name],
                )

                if legend_loss[curve_name]:
                    legend_loss[curve_name] = False

                fig_loss.add_trace(curve, row=row_dic[p], col=col_dic[n])

                # selected features
                group["n_selected"] = group.apply(
                    lambda row: count_selected(row["selected"]), axis=1
                )
                group_n = group[["lamb", "alpha", "n_selected", "name"]]
                avg = group_n.groupby(["lamb", "alpha", "name"]).mean()
                avg = avg.reset_index()
                n_lamb = len(avg["lamb"].unique())
                n_alpha = len(avg["alpha"].unique())
                z = (
                    avg.sort_values(["lamb", "alpha"])["n_selected"]
                    .values.reshape(n_lamb, n_alpha, order="C")
                    .T
                )
                fig3 = go.Surface(
                    z=z,
                    x=np.log(avg["lamb"].unique()),
                    y=avg["alpha"].unique(),
                    showscale=False,
                )

                fig_am[group["name"].unique()[0]].add_trace(
                    fig3, row=row_dic[p], col=col_dic[n]
                )

            # alpha-FDR figure
            title = f"Dataset: {mapping_data_name[data]}"
            fig.update_layout(
                template="ggplot2",
                legend_title_text="Association measure:",
                title={"text": title, "x": 0.85, "y": 0.88},
                font=dict(size=22),
            )

            fig.layout.annotations[-2]["font"] = {"size": 30}
            fig.layout.annotations[-1]["xshift"] -= 15
            fig.layout.annotations[-1]["font"] = {"size": 22}
            fig.update_yaxes(range=(-0.50, 1.0), tickvals=tikz_y, ticktext=tikz_text_y)
            tikz_x = list(log_scale)
            tikz_text_x = [f"{el}" for el in list(x)]
            tikz_text_x[0] = "0"
            fig.update_xaxes(
                range=(log_scale.min(), log_scale.max()),
                tickvals=tikz_x,
                ticktext=tikz_text_x,
            )
            fig.update_layout(legend=dict(x=0.75, y=0.95))

            model_name = mapping_data_name[data].replace(".", "_")
            basename = f"alpha_fdr/{model_name}_lambda_controls_{opti}_{penal}"
            fig.write_image(
                f"{basename}.png",
                width=1350,
                height=900,
            )
            fig.write_html(f"{basename}.html")

            # loss figure
            title = f"Dataset: {mapping_data_name[data]}"
            fig_loss.update_layout(
                template="ggplot2",
                legend_title_text="Association measure:",
                title={"text": title, "x": 0.85, "y": 0.88},
                font=dict(size=22),
            )

            fig_loss.layout.annotations[-2]["font"] = {"size": 30}
            fig_loss.layout.annotations[-1]["xshift"] -= 15
            fig_loss.layout.annotations[-1]["font"] = {"size": 22}
            tikz_x = list(log_scale)
            tikz_text_x = [f"{el}" for el in list(x)]
            tikz_text_x[0] = "0"
            fig_loss.update_xaxes(
                range=(log_scale.min() - 0.1, log_scale.max() + 0.1),
                tickvals=tikz_x,
                ticktext=tikz_text_x,
            )
            fig_loss.update_layout(legend=dict(x=0.75, y=0.95))

            model_name = mapping_data_name[data].replace(".", "_")
            basename = f"loss/{model_name}_lambda_controls_loss_{opti}_{penal}"
            fig_loss.write_image(
                f"{basename}.png",
                width=1350,
                height=900,
            )
            fig_loss.write_html(f"{basename}.html")

            for el in table_data["name"].unique():

                if not os.path.isdir(f"selected_features/{el}"):
                    os.mkdir(f"selected_features/{el}")
                title = f"Dataset: {mapping_data_name[data]}"
                fig_am[el].update_layout(
                    template="ggplot2", title={"text": title, "x": 0.85, "y": 0.88}
                )
                fig_am[el].update_scenes(
                    xaxis_title_text="lambda",
                    yaxis_title_text="alpha",
                    zaxis_title="Selected features",
                    xaxis=dict(ticktext=tikz_text_x, tickvals=tikz_x),
                )
                basename = (
                    f"selected_features/{el}/{model_name}_selected_{opti}_{penal}"
                )
                fig_am[el].write_image(
                    f"{basename}.png",
                    width=1350,
                    height=900,
                )
                fig_am[el].write_html(f"{basename}.html")

                # tikz_x = list(log_scale)
                # tikz_text_x = [f"{el}" for el in list(x)]
                # tikz_text_x[0] = "0"
                # fig3.update_layout(title='Selected features', autosize=False,
                #   width=500, height=500,
                #         ticktext= tikz_text_x,
                #         tickvals= tikz_x)))
                # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
