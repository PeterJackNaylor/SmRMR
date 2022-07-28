#!/usr/bin/env python
"""
Input variables:
    - FILE: path containing all simulated data
Output files:
    - *.png files
"""
import os
from colors import mapping_data_name

from utils_plot import (
    read_and_prep_data,
    make_subplot_fn,
    add_2d_plot,
    decorate_and_save,
)
from nice_tables import alpha2_table

alpha_html = "<i>&#945;</i>"

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

tikz_y = [0, 0.25, 0.5, 0.75, 1.0]
tikz_text_y = ["0.00", "0.25", "0.50", "0.75", "1.00"]

tikz_x = [0.25, 0.5, 0.75]
tikz_text_x = ["0.25", "0.50", "0.75"]

id_ = [i / 100 for i in range(0, 105, 5)]


def main():
    # Load data
    grouping_var = ["optimizer", "penalty"]
    table, grouped, datasets = read_and_prep_data(
        "${FILE}", grouping=grouping_var, deal_with_None_penalty=False
    )

    for (opti, penal), sub_table in grouped:

        for data in list(datasets):
            table_data = sub_table.loc[sub_table["run"] == data]

            groups = table_data.groupby(["n", "p", "AM", "kernel"])
            fig_fdr, legend_fdr = make_subplot_fn(titles, alpha_html, "FDR")

            for g_n, group in groups:

                n = int(g_n[0])
                p = int(g_n[1])
                name = g_n[2]
                kernel = g_n[3]

                y_var = "value"
                vars_group = ["alpha"]
                fig_fdr, legend_fdr = add_2d_plot(
                    group,
                    n,
                    p,
                    name,
                    kernel,
                    y_var,
                    vars_group,
                    fig_fdr,
                    legend_fdr,
                    legendrule="once",
                    log_scale=False,
                    add_abscisse=False,
                    add_identity=True,
                )

            model_name = mapping_data_name[data].replace(".", "_")
            # FDR figure
            if not os.path.isdir("fdr"):
                os.mkdir("fdr")
            basename = f"fdr/{model_name}_fdr_controls_{opti}_{penal}"
            decorate_and_save(
                fig_fdr,
                model_name,
                None,
                False,
                basename,
                tikz_y=tikz_y,
                tikz_text_y=tikz_text_y,
                y_range=(0, 1.0),
                tikz_x=tikz_x,
                tikz_text_x=tikz_text_x,
                x_range=(0, 1.0),
            )

    # alpha 2 check, alpha 2 depends only on the measure, kernel and data
    alpha2_values = alpha2_table(table)
    alpha2_values.to_csv("alpha2_values.csv")


if __name__ == "__main__":
    main()
