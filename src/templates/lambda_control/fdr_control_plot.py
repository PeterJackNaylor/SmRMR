import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import ast
import plotly.express as px

data_color = {
    "linear_0": "#8dd3c7",
    "linear_00": "#ffffb3",
    "linear_1": "#bebada",
    "linear_2": "#fb8072",
    "categorical_0": "#80b1d3",
    "categorical_1": "#fdb462",
    "categorical_2": "#b3de69",
    "nonlinear_1": "#fccde5",
    "nonlinear_2": "#d9d9d9",
    "nonlinear_3": "#bc80bd",
    "nonlinear_4": "#ccebc5",
    "nonlinear_5": "#ffed6f",
}


def get_row_col_pos(run_n_p, row):
    name, n, p = run_n_p
    # row_pos = [el for el in range(len(row)) if row[el] == name][0]
    row_pos = [el for el in range(len(row)) if row[el] == [n, p]][0]
    return name, row_pos + 1


def count(row):

    selected = ast.literal_eval(row)
    return len(selected)


def fdr_plot(table):
    run0, n0, p0, alpha0 = table.loc[0, ["run", "n", "p", "alpha"]]
    n = table[
        (table["run"] == run0)
        & (table["n"] == n0)
        & (table["p"] == p0)
        & (table["alpha"] == alpha0)
    ].shape[0]
    sub_table = table.groupby(["run", "n", "p", "alpha"]).mean()
    sub_table_sd = (table.groupby(["run", "n", "p", "alpha"]).std())[["value"]]
    sub_table_sd.columns = ["sd_value"]
    sub_table_sd = sub_table_sd["sd_value"] * 1.96 / n**0.5
    sub_table = sub_table.join(sub_table_sd).groupby(["run", "n", "p"])

    # rows = table["run"].unique().tolist()
    rows = (
        table.groupby(["n", "p"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})[["n", "p"]]
        .values.tolist()
    )
    # rows.sort()
    rows = sorted(rows, key=lambda x: x[1])
    row_titles = ["({}, {})".format(el[0], el[1]) for el in rows]
    # row_titles = ['1', '2']
    fig = make_subplots(
        rows=len(rows),
        cols=3,
        start_cell="bottom-left",
        column_titles=("Alpha vs FDR", "# parameters", "No returns"),
        row_titles=row_titles,
    )
    for run_n_p, t in sub_table:
        t = t.reset_index()
        name, row_pos = get_row_col_pos(run_n_p, rows)
        showlegend = row_pos == 1
        fig.add_trace(
            go.Scatter(
                x=t["alpha"],
                y=t["value"],
                name=name,
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates
                    array=t["sd_value"],
                    visible=True,
                ),
                marker=dict(size=8, color=data_color[name]),
                legendgroup=name,
                showlegend=showlegend,
            ),
            row=row_pos,
            col=1,
        )
        line = np.arange(0, 1, 0.01)

        fig.add_trace(
            go.Scatter(
                x=line,
                y=line,
                marker=dict(
                    size=16,
                    color="black",  # set color equal to a variable
                ),
                showlegend=False,
            ),
            row=row_pos,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=t["alpha"],
                y=t["N_selected"],
                name=name,
                marker_color=data_color[name],
                legendgroup=name,
                showlegend=False,
            ),
            row=row_pos,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=t["alpha"],
                y=t["void"],
                name=name,
                marker_color=data_color[name],
                legendgroup=name,
                showlegend=False,
            ),
            row=row_pos,
            col=3,
        )
    fig.update_layout(
        showlegend=True,
        font=dict(
            size=18,
        ),
    )
    # fig.show()
    fig.write_image("lambda_control.png", width=1350, height=900)
    fig.write_html("lambda_control.html")


def ms_kern(row):
    ms = row["MS"]
    kern = row["kernel"]
    if ms == "HSIC":
        return f"{ms}({kern})"
    else:
        return ms


def main():
    table = pd.read_csv("performance.tsv", sep="\t")
    table.loc[table["value"] == -1, "value"] = 0
    table["N_selected"] = table.selected.apply(lambda row: count(row))
    table["void"] = table.N_selected.apply(lambda row: int(row > 0))

    fdr_plot(table)

    sub_table = table[table["alpha"] == table["alpha"].values[0]]

    sub_table["ms_kern"] = sub_table.apply(lambda row: ms_kern(row), axis=1)
    sub_table = sub_table.groupby(["run", "ms_kern"]).size().reset_index()
    sub_table.columns = ["dataset", "MS", "Count"]
    sub_table.sort_values(by=["dataset", "MS"])
    fig = px.bar(
        sub_table, x="dataset", y="Count", color="MS", title="Pick association measures"
    )
    fig.write_image("measure_stat.png", width=1350, height=900)
    fig.write_html("measure_stat.html")

    sub_table = table[table["alpha"] == table["alpha"].values[0]]

    sub_table = sub_table.groupby(["run", "penalty"]).size().reset_index()
    sub_table.columns = ["dataset", "Penalty", "Count"]
    sub_table.sort_values(by=["dataset", "Penalty"])
    fig = px.bar(
        sub_table,
        x="dataset",
        y="Count",
        color="Penalty",
        title="Picked penalty measures",
    )
    fig.write_image("penalty.png", width=1350, height=900)
    fig.write_html("penalty.html")

    sub_table = table[table["alpha"] == table["alpha"].values[0]]

    sub_table["ms_kern"] = sub_table.apply(lambda row: ms_kern(row), axis=1)
    sub_table.sort_values(by=["run", "penalty", "ms_kern"])

    fig = px.box(sub_table, x="run", y="lamb", color="penalty", log_y=True)

    fig.write_image("lambda_vs_penalty.png", width=1350, height=900)
    fig.write_html("lambda_vs_penalty.html")

    fig = px.box(sub_table, x="run", y="lamb", color="ms_kern", log_y=True)
    fig.write_image("lambda_vs_ms.png", width=1350, height=900)
    fig.write_html("lambda_vs_ms.html")


if __name__ == "__main__":
    main()
