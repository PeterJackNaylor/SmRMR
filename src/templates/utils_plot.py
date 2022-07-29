import os

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from colors import color_dictionary_fdr, color_dictionary_fdr_keys, name_mapping_fdr

row_dic = {100: 1, 500: 2, 5000: 3}
col_dic = {100: 1, 500: 2, 1000: 3}


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


def read_and_prep_data(file_name, grouping, deal_with_None_penalty=True):

    # Load data
    # table = pd.read_csv("${FILE}", sep="\t")
    # set to 0 those which didn't select anything

    table = pd.read_csv(file_name, sep="\t")
    table.loc[table["value"] == -1, "value"] = 0
    table["kernel"] = table["kernel"].fillna("unspecified")
    table = table.dropna()
    table = table.loc[table["run"] != 0]

    if deal_with_None_penalty:
        if_none_0 = "None" in table["penalty"].unique()
        table.loc[table["penalty"] == "None", "lamb"] = table["lamb"].min() / 10

        tmp_pen = [el for el in table["penalty"].unique() if el != "None"]
        tmp_tab = table[table["penalty"] == "None"]
        for pen in tmp_pen:
            tmp_tab.loc[:, "penalty"] = pen
            table = pd.concat([table, tmp_tab])
        table = table[table["penalty"] != "None"]

    datasets = np.unique(table["run"])
    table["name"] = table.apply(lambda row: new_name(row["AM"], row["kernel"]), axis=1)
    grouped = table.groupby(grouping)
    if deal_with_None_penalty:
        return table, grouped, datasets, if_none_0
    else:
        return table, grouped, datasets


def make_subplot_fn(
    titles, x_axis_n, y_axis_n, color_dictionary=color_dictionary_fdr_keys
):
    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        subplot_titles=titles,
        x_title=x_axis_n,  # , 'font': {'size': 0}},
        y_title=y_axis_n,
    )
    legend = {el: True for el in color_dictionary}
    return fig, legend


def make_subplot_isoline(titles, x_axis_n, y_axis_n, unique_models):
    def g():
        return make_subplots(
            rows=3,
            cols=3,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.06,
            horizontal_spacing=0.04,
            subplot_titles=titles,
            x_title=x_axis_n,  # , 'font': {'size': 0}},
            y_title=y_axis_n,
        )

    fig_isoline = {el: g() for el in unique_models}
    return fig_isoline


def make_subplot_3d(titles, unique_models):
    def g():
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

    fig_3d = {el: g() for el in unique_models}
    return fig_3d


def add_2d_plot(
    data,
    n,
    p,
    name,
    kernel,
    y_var,
    grouping_var,
    fig,
    legend,
    legendrule="once",
    log_scale=True,
    add_abscisse=True,
    add_identity=False,
    scale_by_max=False,
):

    if scale_by_max:
        data[y_var] = data[y_var] / data[y_var].abs().max()
    grouped_by_var = data.groupby(grouping_var)
    mean = grouped_by_var.mean()
    sample_number = grouped_by_var.count()
    std = grouped_by_var.std()

    x = mean.index.sort_values()
    y = mean.loc[x, y_var]
    # if scale_by_max:
    #     y = y / np.absolute(y.max())
    std_y = std.loc[x, y_var]
    n_samples = sample_number.loc[x, y_var]
    err = 1.96 * std_y / (n_samples) ** 0.5
    curve_name = name_mapping_fdr(name, kernel)
    if log_scale:
        x = np.log(x)
    if legendrule == "once":
        showlegend = legend[curve_name]
    else:
        showlegend = False
    curve = go.Scatter(
        x=x,
        y=y,
        name=curve_name,
        error_y=dict(array=err),
        marker=dict(
            color=color_dictionary_fdr(
                name,
                kernel,
            )
        ),
        showlegend=showlegend,
    )
    if legendrule == "once" and legend[curve_name]:
        legend[curve_name] = False

    fig.add_trace(curve, row=row_dic[p], col=col_dic[n])
    if add_abscisse:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[0 for el in list(x)],
                name="",
                marker={"color": "rgb(0, 0, 0)"},
                line=dict(width=0.5),
                showlegend=False,
            ),
            row=row_dic[p],
            col=col_dic[n],
        )

    if add_identity:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=x,
                name="",
                mode="lines",
                marker={"color": "rgb(0, 0, 0)"},
                line=dict(width=1.5),
                showlegend=False,
            ),
            row=row_dic[p],
            col=col_dic[n],
        )

    if legendrule == "once":
        return fig, legend
    else:
        return fig


def compute_z_selected(data, fn, var, list_var, logscale=True):
    # selected features, 3D plots and isoline plot
    data["n_selected"] = data.apply(lambda row: fn(row[var]), axis=1)
    group_n = data[list_var + ["n_selected"]]
    avg = group_n.groupby(list_var).mean()
    avg = avg.reset_index()
    n_lamb = len(avg[list_var[0]].unique())
    n_alpha = len(avg[list_var[1]].unique())
    z = (
        avg.sort_values(list_var[:2])["n_selected"]
        .values.reshape(n_lamb, n_alpha, order="C")
        .T
    )
    x = avg[list_var[0]].unique()
    if logscale:
        x = np.log(x)
    y = avg[list_var[1]].unique()
    return z, x, y


def add_3d_plot(fig_3d, p, n, z, x, y, name):
    fig = go.Surface(
        z=z,
        x=x,
        y=y,
        showscale=False,
    )

    fig_3d[name].add_trace(fig, row=row_dic[p], col=col_dic[n])
    return fig_3d


def add_iso_plots(fig, p, n, z, x, y, name):
    fig_iso = go.Contour(
        z=z,
        x=x,
        y=y,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
            #
        ),
        line=dict(width=2, color="white"),
    )
    fig[name].add_trace(fig_iso, row=row_dic[p], col=col_dic[n])
    fig[name].update_traces(showscale=False)
    return fig


def cut_one(el):
    el = str(np.exp(float(el)))
    for i, digit in enumerate(el):
        if digit == "1":
            break
    return el[: i + 1]


def decorate_and_save(
    fig,
    data_name,
    x,
    if_none_0,
    basename,
    tikz_y=None,
    tikz_text_y=None,
    y_range=(-0.50, 1.0),
    tikz_x=None,
    tikz_text_x=None,
    x_range=None,
    extra_spacing_x=False,
    log_scale=True,
):

    title = f"Dataset: {data_name.replace('_', '.')}"
    fig.update_layout(
        template="ggplot2",
        legend_title_text="Association measure:",
        title={"text": title, "x": 0.85, "y": 0.88},
        font=dict(size=22),
    )
    fig.layout.annotations[-2]["font"] = {"size": 30}
    fig.layout.annotations[-1]["xshift"] -= 15
    fig.layout.annotations[-1]["font"] = {"size": 22}
    if tikz_y is not None:
        fig.update_yaxes(range=y_range, tickvals=tikz_y, ticktext=tikz_text_y)
    if tikz_x is None:
        tikz_x = list(x)
        tikz_text_x = [f"{el}" for el in list(x)]
        if log_scale:
            tikz_text_x = [f"{cut_one(el)}" for el in tikz_text_x]
        if if_none_0:
            tikz_text_x[0] = "0"
        x_range = (x.min(), x.max())

    if extra_spacing_x:
        x_range = (x_range[0] - 0.1, x_range[1] + 0.1)

    fig.update_xaxes(
        range=x_range,
        tickvals=tikz_x,
        ticktext=tikz_text_x,
    )
    fig.update_layout(legend=dict(x=0.75, y=0.95))
    fig.write_image(
        f"{basename}.png",
        width=1350,
        height=900,
    )
    fig.write_html(f"{basename}.html")


def decorate_3d_mult_and_save(
    fig,
    data_name,
    xname,
    yname,
    zname,
    unique_models,
    pathfolder,
    basename,
    x,
    if_none_0,
    log_scale=True,
):
    title = f"Dataset: {data_name.replace('_', '.')}"

    for el in unique_models:
        if not os.path.isdir(f"{pathfolder}/{el}"):
            os.mkdir(f"{pathfolder}/{el}")
        fig[el].update_layout(
            template="ggplot2", title={"text": title, "x": 0.85, "y": 0.88}
        )
        tikz_x = list(x)
        tikz_text_x = [f"{el}" for el in list(x)]
        if log_scale:
            tikz_text_x = [f"{cut_one(el)}" for el in tikz_text_x]
        if if_none_0:
            tikz_text_x[0] = "0"
        fig[el].update_scenes(
            xaxis_title_text=xname,
            yaxis_title_text=yname,
            zaxis_title=zname,
            xaxis=dict(ticktext=tikz_text_x, tickvals=tikz_x),
        )
        basename_f = f"selected_features/{el}/{basename}"
        fig[el].write_image(
            f"{basename_f}.png",
            width=1350,
            height=900,
        )
        fig[el].write_html(f"{basename_f}.html")


def decorate_multi_and_save(
    fig,
    data_name,
    xname,
    yname,
    zname,
    unique_models,
    pathfolder,
    basename,
    x,
    if_none_0,
    log_scale=True,
):
    title = f"Dataset: {data_name.replace('_', '.')}"

    for el in unique_models:
        if not os.path.isdir(f"{pathfolder}/{el}"):
            os.mkdir(f"{pathfolder}/{el}")
        # isoline figures
        fig[el].update_layout(
            template="ggplot2", title={"text": title, "x": 0.85, "y": 0.88}
        )
        tikz_x = list(x)
        tikz_text_x = [f"{el}" for el in list(x)]
        if log_scale:
            tikz_text_x = [f"{cut_one(el)}" for el in tikz_text_x]
        if if_none_0:
            tikz_text_x[0] = "0"
        fig[el].update_xaxes(
            tickvals=tikz_x,
            ticktext=tikz_text_x,
        )
        fig[el].update_scenes(
            xaxis_title_text=xname,
            yaxis_title_text=yname,
            # xaxis=dict(ticktext=tikz_text_x, tickvals=tikz_x),
        )
        basename_f = f"selected_features/{el}/{basename}"
        fig[el].write_image(
            f"{basename_f}.png",
            width=1350,
            height=900,
        )
        fig[el].write_html(f"{basename_f}.html")
