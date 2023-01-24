import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import ast

data_color = {
    "linear_0": "#7fc97f",
    "linear_00": "#beaed4",
    "linear_1": "#ffff99",
    "linear_2": "#386cb0",
    "categorical_0": "#fdc086",
    "categorical_1": "#f0027f",
    "categorical_2": "#bf5b17",
}


def get_row_col_pos(run_n_p, row):
    name, n, p = run_n_p
    # row_pos = [el for el in range(len(row)) if row[el] == name][0]
    row_pos = [el for el in range(len(row)) if row[el] == [n, p]][0]
    return name, row_pos + 1


def count(row):

    selected = ast.literal_eval(row)
    return len(selected)


def main():
    table = pd.read_csv("performance.tsv", sep="\t")
    table.loc[table["value"] == -1, "value"] = 0
    table["N_selected"] = table.selected.apply(lambda row: count(row))
    table["void"] = table.N_selected.apply(lambda row: int(row > 0))
    sub_table = (
        table.groupby(["run", "n", "p", "alpha"]).mean().groupby(["run", "n", "p"])
    )
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
                marker=dict(size=16, color=data_color[name]),
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
    fig.show()
    fig.write_image("lambda_control.png", width=1350, height=900)
    fig.write_html("lambda_control.html")


if __name__ == "__main__":
    main()
