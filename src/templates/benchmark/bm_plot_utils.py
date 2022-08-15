import pandas as pd
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

table = pd.read_csv(sys.argv[1], sep="\t")


def f(x):
    dic = {}
    param = x.split(";")
    for key_value in param:
        key, value = key_value.split("=")
        dic[key] = value

    data, np_ = dic["data"].split("(")
    n, p = np_[:-1].split(",")
    model = dic["model"]

    if model == "DCLasso":
        params = dic["params"]
    else:
        params = ""

    if "feature_selection" in dic:
        feature_selection = dic["feature_selection"]
    else:
        feature_selection = ""

    return [model, data, n, p, params, feature_selection]


table[["model", "data", "n", "p", "params", "feature_selection"]] = table.run.apply(
    lambda x: f(x)
).to_list()


def inv(x):
    if x["metric"] in ["fpr", "fpr_causal"]:
        metric = f"1-{x['metric']}"
        value = 1 - x["value"]
    else:
        metric = x["metric"]
        value = x["value"]
    return value, metric


table[["value", "metric"]] = table.apply(lambda x: inv(x), axis=1).to_list()


colors = {
    "DCLasso[(scad,DC,linear)]": "rgb(127,0,0)",
    "DCLasso[(None,DC,linear)]": "rgb(215,48,31)",
    "DCLasso[(l1,DC,linear)]": "rgb(252,141,89)",
    "DCLasso[(mcp,DC,linear)]": "rgb(253,212,158)",
    "DCLasso[(scad,PC,linear)]": "rgb(2,56,88)",
    "DCLasso[(None,PC,linear)]": "rgb(5,112,176)",
    "DCLasso[(l1,PC,linear)]": "rgb(116,169,207)",
    "DCLasso[(mcp,PC,linear)]": "rgb(166,189,219)",
    "DCLasso[(mcp,HSIC,linear)]": "rgb(77,0,75)",
    "DCLasso[(scad,HSIC,linear)]": "rgb(129,15,124)",
    "DCLasso[(None,HSIC,linear)]": "rgb(136,65,157)",
    "DCLasso[(l1,HSIC,linear)]": "rgb(140,107,177)",
    "DCLasso[(mcp,HSIC,gaussian)]": "rgb(140,150,198)",
    "DCLasso[(scad,HSIC,gaussian)]": "rgb(158,188,218)",
    "DCLasso[(None,HSIC,gaussian)]": "rgb(191,211,230)",
    "DCLasso[(l1,HSIC,gaussian)]": "rgb(247,104,161)",
    "DCLasso[(mcp,HSIC,sigmoid)]": "rgb(221,52,151)",
    "DCLasso[(scad,HSIC,sigmoid)]": "rgb(250,159,181)",
    "DCLasso[(None,HSIC,sigmoid)]": "rgb(174,1,126)",
    "DCLasso[(l1,HSIC,sigmoid)]": "rgb(250,159,181)",
    "knn[hsic_lasso]": "rgb(65,171,93)",
    "logistic_regression[hsic_lasso]": "rgb(35,139,69)",
    "svc[hsic_lasso]": "rgb(0,109,44)",
    "random_forest[hsic_lasso]": "rgb(0,68,27)",
    "stg": "rgb(116,196,118)",
}

print("group table by datasets")
metrics = []
y = []


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


def figure_out_name(model, params, feature_selection):
    name = f"{model}"
    if feature_selection != "":
        name += f"[{feature_selection}]"
    if params != "":
        name += f"[{params}]"
    return name


def plotlytt():
    for data_cat, data_by_data in table.groupby("data"):
        fig = go.Figure()
        # shortname =
        fig = make_subplots(
            rows=3,
            cols=3,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.06,
            horizontal_spacing=0.04,
            subplot_titles=titles,
            x_title="models",  # , 'font': {'size': 0}},
            y_title="scores",
        )
        for_saving = pd.pivot_table(
            data_by_data,
            index=["run"],
            columns=["metric"],
            values=["value"],
            aggfunc="mean",
        )
        for_saving.columns = [el[1] for el in for_saving.columns]
        helpers = (
            data_by_data[["n", "p", "params", "model", "run"]].groupby("run").first()
        )
        for_saving = for_saving.join(helpers)

        for_saving.to_csv(f"results_{data_cat}.csv")
        groups = data_by_data.groupby(
            ["n", "p", "model", "params", "feature_selection"]
        )
        legend = []

        for g_n, group in groups:
            n, p, model, params, feature_selection = g_n
            n, p = int(n), int(p)

            y = group["value"]
            x = group["metric"]
            name = figure_out_name(model, params, feature_selection)
            print(name)
            if name in legend:
                display_legend = False
            else:
                display_legend = True
                legend.append(name)

            color = colors[name]
            boxes = go.Box(
                y=y,
                x=x,
                name=name,
                marker_color="black",
                fillcolor=color,
                showlegend=False,
                boxpoints="suspectedoutliers",
                boxmean=True,
                # boxpoints=False,
                marker_size=2,
            )
            fig.add_trace(boxes, row=row_dic[p], col=col_dic[n])
            if display_legend:
                boxes = go.Scatter(
                    x=[None],
                    y=[None],
                    name=name,
                    mode="markers",
                    marker_color=color,
                    marker=dict(size=12, line=dict(width=2, color="black")),
                    marker_symbol="square",
                    showlegend=True,
                    marker_size=15,
                )

                fig.add_trace(boxes, row=row_dic[p], col=col_dic[n])

        fig.update_layout(
            boxmode="group",
            title="Comparing different methods on the simulated datasets",
            xaxis_tickfont_size=14,
            yaxis=dict(
                title="USD (millions)",
                titlefont_size=16,
                tickfont_size=14,
            ),
            legend=dict(
                x=0.75,
                y=0.95,
                bgcolor="rgba(255, 255, 255, 0)",
                bordercolor="rgba(255, 255, 255, 0)",
            ),
            barmode="group",
            bargap=0.15,  # gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # gap between bars of the same location coordinate.
        )
        fig.show()


def f_run(line):
    model, fs, data = line.split(";")
    model = model.split("=")[1]
    fs = fs.split("=")[1]
    data = data.split("=")[1]
    datatype = data.split("(")[0]
    n, p = data[:-1].split("(")[1].split(",")
    mode = "regression" if "linear" in datatype else "classification"

    return model, fs, data, n, p, mode, datatype


def process(table):
    table[["model", "fs", "data", "n", "p", "mode", "datatype"]] = table.apply(
        lambda row: f_run(row["run"]), axis=1, result_type="expand"
    )
    return table


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


def create_table(df, mode):
    if mode == "regression":
        order_cols = ["mse", "fdr_causal", "fpr_causal", "tpr_causal"]
        nicer_names = ["MSE", "FDR", "FPR", "TPR"]
    else:
        order_cols = ["accuracy", "fdr_causal", "fpr_causal", "tpr_causal"]
        nicer_names = ["Acc", "FDR", "FPR", "TPR"]

    pivot_df = pd.pivot_table(
        df,
        index=["datatype", "model", "fs"],
        columns=["metric"],
        values=["value"],
        aggfunc="mean",
    )
    new_cols = [it[1] for it in pivot_df.columns]
    pivot_df.columns = new_cols
    pivot_df = pivot_df[order_cols]
    pivot_df.columns = nicer_names
    return pivot_df


def multi_table_plot(df):

    modes = df["mode"].unique()
    for mode in modes:
        # fig = make_subplots(cols=3, rows=3, shared_xaxes=False, shared_yaxes=False)
        df_mode = df[df["mode"] == mode]
        unique_n_p = df_mode.groupby(["n", "p"]).size().reset_index(name="Freq")
        for i in unique_n_p.index:
            n, p = unique_n_p.loc[i, ["n", "p"]]
            df_n_p_mode = df[(df["n"] == n) & (df["p"] == p)]
            _ = create_table(df_n_p_mode, mode)

            _ = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=list(df.fig_table),
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=[df.Rank, df.State, df.Postal, df.Population],
                            fill_color="lavender",
                            align="left",
                        ),
                    )
                ]
            )


def main(table):

    table = process(table)
    multi_table_plot(table)
