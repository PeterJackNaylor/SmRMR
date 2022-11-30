from utils_plot import read_and_prep_data

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import ast


def unique_double(df, x_var, y_var):
    tmp = df.groupby([x_var, y_var]).size().reset_index().rename(columns={0: "count"})
    tmp = tmp.sort_values([x_var, y_var], ascending=[True, True])
    n_p = [list(el) for el in list(np.array(tmp[[x_var, y_var]]))]
    return n_p


maping_col = {
    "PC": "#a6cee3",
    "HSIC (gaussian)": "#b2df8a",
    "HSIC (linear)": "#fdbf6f",
    "HSIC (inverse-M)": "#fb9a99",
}

maping_symb = {
    "l1": "square",
    "scad": "circle",
    "mcp": "diamond",
}


def add_curve_plot(fig, df, x_var, y_var, i, j, last_i=-1, log_scale=False):
    pen_name = unique_double(df, "penalty", "name")

    x_y_groups = (
        df.groupby(["name", "penalty", x_var])
        .agg({y_var: ["mean", "std"]})
        .reset_index()
    )
    x_y_groups.columns = ["name", "penalty", "lamb", "mean", "std"]

    showLegend = i == j == 0

    for (pen, name) in pen_name:
        tmp_groups = x_y_groups[(x_y_groups.name == name) & (x_y_groups.penalty == pen)]
        tmp_groups = tmp_groups.sort_values([x_var], ascending=[True])
        tmp_fig = go.Scatter(
            x=tmp_groups[x_var],
            y=tmp_groups["mean"],
            mode="lines+markers",
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=tmp_groups["std"],
                visible=True,
            ),
            line=dict(color=maping_col[name], width=4),
            marker=dict(symbol=maping_symb[pen], line_width=4),
            legendgroup=name + pen,  # this can be any string, not just "group"
            name=f"{name} + {pen}",
            showlegend=showLegend,
        )
        fig.add_trace(tmp_fig, row=i + 1, col=j + 1)

    if i == last_i - 1:
        n, p = df.n.unique()[0], df.p.unique()[0]
        fig.update_xaxes(title_text=f"n={n}; p={p}", row=i + 1, col=j + 1)
    if j == 0:
        data = df.run.unique()[0]
        fig.update_yaxes(title_text=f"{data}", row=i + 1, col=j + 1)
    if log_scale:
        idx = x_y_groups[x_var] != 0
        second_small = np.log(x_y_groups.loc[idx, x_var].min() / 10) / 2.30
        maxi = np.log(2 * x_y_groups[x_var].max()) / 2.30
        fig.update_xaxes(type="log", range=[second_small, maxi])

    # fig.update_xaxes(type="log", range=[0,5]) # log range: 10^0=1, 10^5=100000
    # fig.update_yaxes(range=[0,100]) # linear range


def main():
    grouping_var = ["optimizer", "penalty"]
    table, grouped, datasets, if_none_0 = read_and_prep_data(
        "performance.tsv", grouping=grouping_var, deal_with_None_penalty=True
    )
    datasets = list(datasets)
    n_p = unique_double(table, "n", "p")

    fdr_alpha = make_subplots(rows=len(datasets), cols=len(n_p))
    n_features = make_subplots(rows=len(datasets), cols=len(n_p))
    t_features = make_subplots(rows=len(datasets), cols=len(n_p))

    for i, data in enumerate(datasets):
        for j, (n, p) in enumerate(n_p):
            table_data = table[(table.run == data) & (table.n == n) & (table.p == p)]

            table_data["FDR-alpha"] = table_data["value"] - table_data["alpha"]
            add_curve_plot(
                fdr_alpha,
                table_data,
                "lamb",
                "FDR-alpha",
                i,
                j,
                last_i=len(datasets),
                log_scale=True,
            )
            table_data = table_data[table_data["alpha"].round(2) == 0.1]
            table_data["Nselected"] = table_data.selected.apply(
                lambda x: len(ast.literal_eval(x))
            )

            add_curve_plot(
                n_features,
                table_data,
                "lamb",
                "Nselected",
                i,
                j,
                last_i=len(datasets),
                log_scale=True,
            )
            table_data["TrueSelected"] = (
                table_data["Nselected"] - table_data["value"] * table_data["Nselected"]
            )
            add_curve_plot(
                t_features,
                table_data,
                "lamb",
                "TrueSelected",
                i,
                j,
                last_i=len(datasets),
                log_scale=True,
            )
    fdr_alpha.write_image("fdr_alpha.png", width=1350, height=900)
    fdr_alpha.write_html("fdr_alpha.html")
    n_features.write_image("n_selected.png", width=1350, height=900)
    n_features.write_html("n_selected.html")
    t_features.write_image("true_features.png", width=1350, height=900)
    t_features.write_html("true_features.html")


if __name__ == "__main__":
    main()
