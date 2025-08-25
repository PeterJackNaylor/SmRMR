
import os
import pandas as pd
import plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_folder(name):
    try:
        os.mkdir(name)
    except:
        pass
create_folder("csv")
create_folder("tex")
create_folder("png")
create_folder("html")
create_folder("png2")
create_folder("html2")

colour_map = {   
        "HSICLasso": "#006d2c",
        "MRMR": "#810f7c", 
        "smrmr(HSIC, Gaussian, None)": "#fcbba1",
        "smrmr(HSIC, Gaussian, L1)": "#fb6a4a",
        "smrmr(HSIC, Gaussian, MCP)": "#cb181d",
        "smrmr(HSIC, Gaussian, SCAD)": "#a50f15",
        # "smrmr(HSIC, laplacian, None)": "#fee391",
        # "smrmr(HSIC, laplacian, l1)": "#fe9929",
        # "smrmr(HSIC, laplacian, mcp)": "#cc4c02",
        # "smrmr(HSIC, laplacian, scad)": "#993404",
        "smrmr(PC, None)": "#c7e9b4",
        "smrmr(PC, L1)": "#41b6c4",
        "smrmr(PC, MCP)": "#225ea8",
        "smrmr(PC, SCAD)": "#253494"
    }


# colour_map = {   
#         "hsic_lasso": "#006d2c",
#         "mrmr": "#810f7c", 
#         "smrmr(gaussian, HSIC, None)": "#fcbba1",
#         "smrmr(gaussian, HSIC, l1)": "#fb6a4a",
#         "smrmr(gaussian, HSIC, mcp)": "#cb181d",
#         "smrmr(gaussian, HSIC, scad)": "#a50f15",
#         "smrmr(linear, HSIC, None)": "#fee391",
#         "smrmr(linear, HSIC, l1)": "#fe9929",
#         "smrmr(linear, HSIC, mcp)": "#cc4c02",
#         "smrmr(linear, HSIC, scad)": "#993404",
#         "smrmr(PC, None)": "#c7e9b4",
#         "smrmr(PC, l1)": "#41b6c4",
#         "smrmr(PC, mcp)": "#225ea8",
#         "smrmr(PC, scad)": "#253494"
#     }


names = {
    "acc": "Acc",
    "mse": "MSE",
    "tpr": "TPR",
    "fpr": "FPR",
    "f1": "F1",
    "tpr_causal": "TPRf",
    "fpr_causal": "FPRf",
    "fdr_causal": "FDRf",
    "n_selected": "Nf",
}

def hh(val):
    if val == "None":
        return 0
    elif val == "L1":
        return 1
    elif val == "MCP":
        return 2
    elif val == "SCAD":
        return 3
    else:
        return 4
    
def ff(val):
    if val == "HSIC":
        return 1
    elif val == "PC":
        return 2
    else:
        return 0

def gg(val):
    if val == "MRMR":
        return 0
    elif val == "HSICLasso":
        return 1
    else:
        return 2

def tuple_sort(my_tup):
    my_tup.sort(key = lambda x: (gg(x[3]), ff(x[0]), hh(x[2])))
    return my_tup

def curve_name(method):
    ms, kernel, penalty, meth = method
    if meth == "smrmr":
        if ms == "PC":
            name = f"smrmr({ms}, {penalty})"
        else:
            name = f"smrmr({ms}, {kernel}, {penalty})"
    else:
        name = meth
    return name

def plot_var(row, col, var, tab, fig, showlegend=False):
    methods = list(tab.groupby(['MS','KERNEL', 'PENALTY', 'METHOD'], dropna=False).size().index)
    methods = tuple_sort(methods)
    methods = [('HSIC', 'Gaussian', 'None', 'smrmr'), ('HSIC', 'Gaussian', 'L1', 'smrmr'), ('HSIC', 'Gaussian', 'MCP', 'smrmr'), ('HSIC', 'Gaussian', 'SCAD', 'smrmr'),  ('None2', 'None2', 'None2', 'MRMR'),  ('None2', 'None2', 'None2', 'HSICLasso'), ('PC', 'None2', 'None', 'smrmr'), ('PC', 'None2', 'L1', 'smrmr'), ('PC', 'None2', 'MCP', 'smrmr'), ('PC', 'None2', 'SCAD', 'smrmr')]
    for method in methods:
        ms, kernel, penalty, meth = method
        name = curve_name(method)
        if kernel != "linear":
            colour = colour_map[name]
            subtab = tab.loc[slice(None), slice(None), slice(None), meth, ms, kernel, penalty]
            subtab = subtab.sort_values("xaxis").reset_index()
            x = subtab["xaxis"]
            y = subtab[f"{var} mean"]
            y_error = subtab[f"{var} std"]
            data = go.Scatter(x=x, y=y, name=name, legendgroup=name, showlegend=showlegend, line=dict(color=colour), error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=y_error * 1.96 / 10,
                visible=True)
            )
            
            fig.add_trace(data,
                        row=row, col=col)
    if var == "MSE" and y.mean() > 10 and row == 1 and col == 1:
        # next(fig.select_yaxes(row=row, col=col)).update(yaxis_type="log")  
        fig.update_yaxes(type="log", row=row, col=col)
  
    return fig



def inverse(val, name):
    if "FDR" in name or "FPR" in name:
        return 1 - val
    else:
        return val

def get_method(st):

    if st == "hsic_lasso" or st == "mrmr":
        if st == "hsic_lasso":
            return "HSICLasso"
        else:
            return "MRMR"
    else:
        return "smrmr"

def get_ms_kernel_penalty(st, method):
    if method == "smrmr":
        infos = st[8:-1].split(",")
        if infos[-1] == "PC":
            ms = "PC"
            kernel = "None2"
            penalty = infos[0]
            if penalty != "None":
                penalty = penalty.upper()
        else:
            kernel = infos[-1].capitalize()
            ms = infos[1]
            penalty = infos[0]
            if penalty != "None":
                penalty = penalty.upper()
    else:
        if st == "hsic_lasso":
            st = "HSICLasso"
        else:
            st = "MRMR"
        ms = "None2"
        kernel = "None2"
        penalty = "None2"
    return ms, kernel, penalty

def parse(row):
    mod, fs, data = row.split(";")
    mod = mod.split("=")[1]
    fs = fs.split("=")[1]
    method = get_method(fs)
    ms, kernel, penalty = get_ms_kernel_penalty(fs, method)
    data = data.split("=")[1]
    data, size = data[:-1].split("(")
    n, p = size.split(",")
    return mod, fs, data, n, p, method, ms, kernel, penalty


def read_table(path):
    t = pd.read_csv(path, sep="\t")
    t[["model", "fs", "name", "n", "p", "METHOD", "MS", "KERNEL", "PENALTY"]] = (
        t["run"].apply(lambda row: parse(row)).tolist()
    )
    t[["n", "p"]] = t[["n", "p"]].astype(int)
    t["metric"] = t["metric"].apply(lambda row: names[row])
    return t

def map_level(df, dct, level=0):
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i==level else names
                      for i, names in enumerate(index.levels)], inplace=True)
dct = {'None':' '}


def prep_table(table):
    table = table.droplevel("xaxis")
    new_index = pd.MultiIndex.from_product([
        [100, 500],
        [100, 500, 5000],
        ["smrmr", "HSICLasso", "MRMR"],
        ["HSIC", "PC", "None2"],
        ["Gaussian", "None2"],
        ["None", "None2", "L1", "MCP", "SCAD"]],
        names=["n", "p", "Method", "AM", "Kernel", "Penalty"]
    )

    # use the new index to reorder the data
    table = table.reindex(new_index).dropna()



    # table = table.astype(float).round(4)
    columns = table.columns
    means = [el for el in columns if "mean" in el]
    normal_col = [el.replace("f mean", "") for el in means]
    normal_col = [el.replace(" mean", "") for el in normal_col]
    std = [el for el in columns if "std" in el]
    tab_mean = table[means].map(lambda x: float_exponent_notation(x))
    tab_mean.columns = normal_col
    tab_std = table[std].map(lambda x: float_exponent_notation(x))
    tab_std.columns = normal_col
    tab = tab_mean + " $\pm$ " + tab_std
    tab = tab.map(lambda x: combine_float_exponent(x))
    return tab

def float_exponent_notation(float_number, precision_digits=2, format_type="e"):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with `precision_digits` digits of
    mantissa precision, printing a normal decimal if an
    exponent isn't necessary.
    """
    if 0.01 < float_number and float_number < 100:
        format_type = "g"
    else:
        format_type = "e"
    if float_number == 0:
        format_type = "g"
    e_float = "{0:.{1:d}{2}}".format(float_number, precision_digits, format_type)
    if "e" not in e_float:
        return "${}$".format(e_float)
    mantissa, exponent = e_float.split("e")
    cleaned_exponent = exponent.strip("+")
    return "${0} \\times 10^{{{1}}}$".format(mantissa, cleaned_exponent)

def combine_float_exponent(str_number):
    if str_number.count("10^{") > 1:
        mean, std = str_number.split(" $\\pm$ ")
        mean_val, mean_exp = mean.split("\\times")
        std_val, std_exp = std.split("\\times")
        if std_exp == mean_exp:
            mean_val = mean_val.replace("$", "$(")
            str_number = mean_val + " \pm " + std_val.replace("$", "") + ") \\times" + mean_exp
    return str_number









table = read_table("performance.tsv")

table = table.loc[(table.KERNEL != "Linear")]

table.loc[(table.n == 100) & (table.p == 100), "xaxis"] = 1
table.loc[(table.n == 100) & (table.p == 500), "xaxis"] = 2
table.loc[(table.n == 100) & (table.p == 5000), "xaxis"] = 3
table.loc[(table.n == 500) & (table.p == 5000), "xaxis"] = 4
table[["n", "p", "MS", "KERNEL", "PENALTY", "METHOD"]] = table[["n", "p", "MS", "KERNEL", "PENALTY", "METHOD"]].fillna("None")

names = list(table.name.unique())
names = ['categorical_0', 'categorical_1', 'categorical_2', 'linear_00', 'linear_0',  'linear_1', 'linear_2', 'nonlinear_1', 'nonlinear_2', 'nonlinear_3']
variables = ["mse", "acc", "tpr", "fpr", "fdr"]
variables_reg = ["MSE", "TPRf", "FPRf", "FDRf", "Nf"]
variables_cat = ["Acc", "TPRf", "FPRf", "FDRf", "Nf"]



for name in names:
    if name == 'categorical_0':
        i = 0
        ncol = 3
        pos = 0
    elif name == "linear_00":
        i = 0
        ncol = 4
        pos = 3
    elif name == "nonlinear_1":
        i = 0
        ncol = 3
        pos = 7
    show_legend = i == 0
    mode_regression = "linear" in name
    if mode_regression:
        variables = variables_reg
    else:
        variables = variables_cat
    var = "MSE" if mode_regression else "Acc"

    if i == 0:
        subplots_titles = [el.capitalize().replace("_", " ") for el in names[pos:pos+ncol]]
        fig2 = make_subplots(rows=5, cols=ncol, start_cell="top-left", subplot_titles=(subplots_titles), vertical_spacing = 0.01)
    
    i += 1

    t = table.loc[(table.name == name)]
    if len(t) != 0:

        index = ["n", "p", "xaxis", "METHOD", "MS", "KERNEL", "PENALTY"]
        out = pd.concat([t[t["metric"] == v].pivot_table(values="value", index=index, columns="metric", aggfunc=u) for v in variables for u in ["mean", "std"]], axis=1)
        # t = t.groupby(index, dropna=False)
        #out = pd.concat([t.describe()[v][["mean", "std"]] for v in variables], axis=1)
        out.columns = [f"{v} {u}"for v in variables for u in ["mean", "std"]]
        fig = make_subplots(rows=2, cols=2, start_cell="top-left", subplot_titles=(var.upper(), "TPR", "FPR", "FDR"))
        fig = plot_var(1, 1, var, out, fig, showlegend=True)
        fig = plot_var(1, 2, "TPRf", out, fig)
        fig = plot_var(2, 1, "FPRf", out, fig)
        fig = plot_var(2, 2, "FDRf", out, fig)
        fig.update_layout(height=1200, width=2000, title_text=f"Data simulation: {name}")
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3, 4],
                ticktext = ['(100, 100)', '(100, 500)', '(100, 5000)', '(500, 5000)']
            ),
            xaxis2 = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3, 4],
                ticktext = ['(100, 100)', '(100, 500)', '(100, 5000)', '(500, 5000)']
            ),
            xaxis3 = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3, 4],
                ticktext = ['(100, 100)', '(100, 500)', '(100, 5000)', '(500, 5000)']
            ),
            xaxis4 = dict(
                tickmode = 'array',
                tickvals = [1, 2, 3, 4],
                ticktext = ['(100, 100)', '(100, 500)', '(100, 5000)', '(500, 5000)']
            )
        )
        fig.write_html(f"html/{name}.html")
        fig.write_image(f"png/{name}.png")

        prep_tab = prep_table(out)
        prep_tab.to_csv(f"csv/{name}.csv")
        with open(f"tex/{name}.tex", "w") as f:
            latex = prep_tab.style.to_latex(caption=f"Detailed benchmark for the data simulation : {name} . Results are displayed with mean value $\pm$ standard deviation computed over 100 trials.".replace("_", " "), label=f"res_{name}", hrules=True, multirow_align="c", clines="all;data", column_format="|c|c|c|c|c|c|c|c|c|c|c|")
            latex = latex.replace("\\toprule", "\\toprule \\toprule")
            latex = latex.replace("\\midrule", "\\midrule \\midrule")
            latex = latex.replace("None2", " ")
            latex = latex.replace("None2", " ")
            latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
            latex = latex.replace("\end{tabular}", "\end{tabular}}")
            f.write(latex)
        

        fig2 = plot_var(1, i, var, out, fig2, showlegend=show_legend)
        fig2 = plot_var(2, i, "TPRf", out, fig2)
        fig2 = plot_var(3, i, "FPRf", out, fig2)
        fig2 = plot_var(4, i, "FDRf", out, fig2)
        fig2 = plot_var(5, i, "Nf", out, fig2)
    if i == ncol:
        naxis = 5 * ncol
        for j in range(1, naxis + 1):
            if j in list(range(naxis - ncol + 1, naxis + 1)):
                fig2['layout'][f'xaxis{j}'].update(
                    tickmode = 'array',
                    tickvals = [1, 2, 3, 4],
                    ticktext = ['(100, 100)', '(100, 500)', '(100, 5000)', '(500, 5000)']
                    )
            else:
                axisname = "xaxis" if j == 1 else f'xaxis{j}'
                #import pdb; pdb.set_trace()
                fig2['layout'][axisname].update(
                    tickmode = 'array',
                    tickvals = [1, 2, 3, 4],
                    ticktext = ["", "", "", ""]
                    )
        fig2.update_yaxes(title_text=var, row=1, col=1)
        fig2.update_yaxes(title_text="TPR", row=2, col=1)
        fig2.update_yaxes(title_text="FPR", row=3, col=1)
        fig2.update_yaxes(title_text="FDR", row=4, col=1)
        fig2.update_yaxes(title_text="N", row=5, col=1)
        fig2.update_layout(height=1200, width=1800, 
                           legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.1,
                                xanchor="left",
                                x=0.1,
                                font=dict(size=18),
                            ))

        fig2.write_html(f"html2/{name.split('_')[0]}.html")
        fig2.write_image(f"png2/{name.split('_')[0]}.png")
        
