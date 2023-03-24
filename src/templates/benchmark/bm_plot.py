import sys
import pandas as pd
from plotly.subplots import make_subplots

# from bm_plot_utils import main
import plotly.graph_objects as go


def parse(row):
    mod, fs, data = row.split(";")
    mod = mod.split("=")[1]
    fs = fs.split("=")[1]
    data = data.split("=")[1]
    data, size = data[:-1].split("(")
    n, p = size.split(",")
    return mod, fs, data, n, p


names = {
    "acc": "Acc",
    "mse": "MSE",
    "tpr": "TPR",
    "fpr": "1-FPR",
    "f1": "F1",
    "tpr_causal": "TPRf",
    "fpr_causal": "1-FPRf",
    "fdr_causal": "1-FDRf",
}


def inverse(val, name):
    if "FDR" in name or "FPR" in name:
        return 1 - val
    else:
        return val


def read_table(path):
    t = pd.read_csv(path, sep="\t")
    t[["model", "fs", "data", "n", "p"]] = (
        t["run"].apply(lambda row: parse(row)).tolist()
    )
    t[["n", "p"]] = t[["n", "p"]].astype(int)
    t["metric"] = t["metric"].apply(lambda row: names[row])
    t["value"] = t.apply(lambda row: inverse(row["value"], row["metric"]), axis=1)
    return t


def create_table(df):
    order_cols = ["Acc", "MSE", "1-FDRf", "1-FPRf", "TPRf"]

    pivot_df = pd.pivot_table(
        df,
        index=["data", "n", "p", "model", "fs"],
        columns=["metric"],
        values=["value"],
        aggfunc="mean",
    )
    new_cols = [it[1] for it in pivot_df.columns]
    pivot_df.columns = new_cols
    pivot_df = pivot_df[order_cols]
    return pivot_df


def create_fig(t):
    t_data = t.groupby("data")
    for data, sub_table in t_data:
        fig = create_data_fig(sub_table, data)
        fig.write_image(
            f"{data}_sim_benchmark.png",
            width=1350,
            height=900,
        )
        fig.write_html(f"{data}_sim_benchmark.html")


ordering = {
    "Acc": 0,
    "MSE": 0,
    "F1": 1,
    "TPR": 2,
    "1-FPR": 3,
    "TPRf": 4,
    "1-FPRf": 5,
    "1-FDRf": 6,
}
colors = {
    "DCLasso(None,PC)": "rgb(198,219,239)",
    "DCLasso(l1,PC)": "rgb(107,174,214)",
    "DCLasso(mcp,PC)": "rgb(33,113,181)",
    "DCLasso(scad,PC)": "rgb(8,48,107)",
    "DCLasso(None,HSIC,laplacian)": "rgb(252,187,161)",
    "DCLasso(l1,HSIC,laplacian)": "rgb(251,106,74)",
    "DCLasso(mcp,HSIC,laplacian)": "rgb(203,24,29)",
    "DCLasso(scad,HSIC,laplacian)": "rgb(103,0,13)",
    "DCLasso(None,HSIC,gaussian)": "rgb(199,233,192)",
    "DCLasso(l1,HSIC,gaussian)": "rgb(116,196,118)",
    "DCLasso(mcp,HSIC,gaussian)": "rgb(35,139,69)",
    "DCLasso(scad,HSIC,gaussian)": "rgb(0,68,27)",
    "hsic_lasso": "rgb(255,255,51)",
    "mrmr": "rgb(247,129,191)",
    "stg": "rgb(166,86,40)",
}


def build_add_plot(t, fig, row, col):  # , metric):
    showLegend = row == col == 1
    # t = t[t["metric"] == metric]
    t["ordering"] = t["metric"].apply(lambda row: ordering[row])
    t = t.sort_values(by=["ordering", "fs"])
    fs = list(t["fs"].unique())
    fs.sort()

    # boxes = px.box(t, x="metric", y="value", color="fs")
    for fs_ in fs:
        sub_t = t[t["fs"] == fs_]
        # sub_t.sort_values(by=['ordering'])
        subx = list(sub_t["metric"].values)
        suby = list(sub_t["value"].values)

        boxes = go.Box(
            y=suby,
            x=subx,
            marker_color=colors[fs_],
            name=fs_,
            legendgroup=fs_,
            showlegend=showLegend,
            boxmean=True,
            marker_size=2,
            offsetgroup=fs_,
        )
        fig.add_trace(boxes, row=row, col=col)

    return fig


def create_data_fig(t, dataname):
    n_p = list(t[["n", "p"]].value_counts().index)
    n_p.sort()
    nrows = len(n_p)
    fig = make_subplots(
        rows=nrows,
        # cols=len(metrics),
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        # subplot_titles=titles,
        x_title="models",  # , 'font': {'size': 0}},
        y_title="scores",
    )

    row = 1
    for n, p in n_p:
        sub_table = t[(t["n"] == n) & (t["p"] == p)]
        col = 1
        fig = build_add_plot(sub_table, fig, row, col)
        # fig.add_trace(fig_col, row=row, col=col)
        # for metric in metrics:
        #     fig_col = build_plot(sub_table, metric)
        #     fig.add_trace(fig_col, row=row, col=col)
        #     col += 1
        row += 1
    fig.update_layout(title_text=dataname, boxmode="group")
    return fig


table = read_table(sys.argv[1])
pivot = create_table(table)
pivot.to_csv("benchmark_results.csv")
create_fig(table)
