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
    count_selected,
    make_subplot_fn,
    make_subplot_isoline,
    # make_subplot_3d,
    add_2d_plot,
    decorate_and_save,
    compute_z_selected,
    # add_3d_plot,
    add_iso_plots,
    # decorate_3d_mult_and_save,
    decorate_multi_and_save,
)
from nice_tables import alpha2_table


def identity(x):
    return x


lambda_html = "<i>&#955;</i>"
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

tikz_y = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
tikz_text_y = ["-0.50", "-0.25", "0.00", "0.25", "0.50", "0.75", "1.00"]

tikz_x = [0.25, 0.5, 0.75]
tikz_text_x = ["0.25", "0.50", "0.75"]

id_ = [i / 100 for i in range(-100, 105, 5)]
flat_line = [0 for el in id_]


def main():
    grouping_var = ["optimizer", "penalty"]
    table, grouped, datasets, if_none_0 = read_and_prep_data(
        "${FILE}", grouping=grouping_var, deal_with_None_penalty=True
    )

    for elements, sub_table in grouped:
        opti, penal = elements
        for data in list(datasets):
            table_data = sub_table.loc[sub_table["run"] == data]

            groups = table_data.groupby(["n", "p", "AM", "kernel", "name"])

            fig_alpha_fdr, legend_alpha_fdr = make_subplot_fn(
                titles, lambda_html, alpha_html + " - FDR"
            )
            fig_loss_train, legend_loss_train = make_subplot_fn(
                titles, lambda_html, "loss (train)"
            )
            fig_loss_valid, legend_loss_valid = make_subplot_fn(
                titles, lambda_html, "loss (validation)"
            )
            fig_R_constraint, legend_R_constraint = make_subplot_fn(
                titles, lambda_html, "R - N1(beta)"
            )
            unique_models = table_data.name.unique()
            fig_isoline = make_subplot_isoline(
                titles, lambda_html, alpha_html, unique_models
            )
            fig_isoline_fdr = make_subplot_isoline(
                titles, lambda_html, alpha_html, unique_models
            )
            # fig_3d_selected_feats = make_subplot_3d(titles, unique_models)

            for g_n, group in groups:

                n = int(g_n[0])
                p = int(g_n[1])
                name = g_n[2]
                kernel = g_n[3]
                vars_group = ["lamb"]
                # alpha - fdr
                y_var = "control_diff"
                group[y_var] = group["alpha"] - group["value"]
                fig_alpha_fdr, legend_alpha_fdr = add_2d_plot(
                    group,
                    n,
                    p,
                    name,
                    kernel,
                    y_var,
                    vars_group,
                    fig_alpha_fdr,
                    legend_alpha_fdr,
                    legendrule="once",
                    log_scale=True,
                    add_abscisse=True,
                )

                # plot loss train vs lambda
                alpha_group = group[group["alpha"] == 0.1]
                y_var = "loss_train"
                fig_loss_train, legend_loss_train = add_2d_plot(
                    alpha_group,
                    n,
                    p,
                    name,
                    kernel,
                    y_var,
                    vars_group,
                    fig_loss_train,
                    legend_loss_train,
                    legendrule="once",
                    log_scale=True,
                    add_abscisse=False,
                    scale_by_max=True,
                )

                # plot loss validation vs lambda
                y_var = "loss_valid"
                fig_loss_valid, legend_loss_valid = add_2d_plot(
                    alpha_group,
                    n,
                    p,
                    name,
                    kernel,
                    y_var,
                    vars_group,
                    fig_loss_valid,
                    legend_loss_valid,
                    legendrule="once",
                    log_scale=True,
                    add_abscisse=False,
                    scale_by_max=True,
                )

                # plot R - norm1
                y_var = "diff_R_norm1"
                alpha_group[y_var] = (
                    alpha_group["R"] - alpha_group["norm_1"]
                ) / alpha_group["R"]
                fig_R_constraint, legend_R_constraint = add_2d_plot(
                    alpha_group,
                    n,
                    p,
                    name,
                    kernel,
                    y_var,
                    vars_group,
                    fig_R_constraint,
                    legend_R_constraint,
                    legendrule="once",
                    log_scale=True,
                    add_abscisse=False,
                )

                # selected features
                z, x, y = compute_z_selected(
                    group, count_selected, "selected", ["lamb", "alpha", "name"]
                )
                # fig_3d_selected_feats = add_3d_plot(
                #     fig_3d_selected_feats, p, n, z, x, y, group["name"].unique()[0]
                # )

                fig_isoline = add_iso_plots(
                    fig_isoline, p, n, z, x, y, group["name"].unique()[0]
                )

                z_fdr, x_fdr, y_fdr = compute_z_selected(
                    group, identity, "value", ["lamb", "alpha", "name"]
                )

                fig_isoline_fdr = add_iso_plots(
                    fig_isoline_fdr,
                    p,
                    n,
                    z_fdr,
                    x_fdr,
                    y_fdr,
                    group["name"].unique()[0],
                )
            model_name = mapping_data_name[data].replace(".", "_")
            if not os.path.isdir("alpha_fdr"):
                os.mkdir("alpha_fdr")
            # alpha-FDR figure
            basename = f"alpha_fdr/{model_name}_lambda_controls_{opti}_{penal}"
            decorate_and_save(
                fig_alpha_fdr,
                model_name,
                x,
                if_none_0,
                basename,
                tikz_y=tikz_y,
                tikz_text_y=tikz_text_y,
                y_range=(-0.50, 1.0),
            )

            # loss train figure
            if not os.path.isdir("loss_train"):
                os.mkdir("loss_train")
            basename = f"loss_train/{model_name}_lambda_controls_loss_{opti}_{penal}"
            decorate_and_save(
                fig_loss_train, model_name, x, if_none_0, basename, extra_spacing_x=True
            )

            # loss valid figure
            if not os.path.isdir("loss_validation"):
                os.mkdir("loss_validation")
            basename = (
                f"loss_validation/{model_name}_lambda_controls_loss_{opti}_{penal}"
            )
            decorate_and_save(
                fig_loss_valid, model_name, x, if_none_0, basename, extra_spacing_x=True
            )

            # R constraint figure
            if not os.path.isdir("R_constraint"):
                os.mkdir("R_constraint")
            basename = f"R_constraint/{model_name}_R_constraint_{opti}_{penal}"
            decorate_and_save(
                fig_R_constraint,
                model_name,
                x,
                if_none_0,
                basename,
                extra_spacing_x=True,
            )

            if not os.path.isdir("selected_features"):
                os.mkdir("selected_features")
            # basename = f"{model_name}_selected_{opti}_{penal}"
            # decorate_3d_mult_and_save(
            #     fig_3d_selected_feats,
            #     model_name,
            #     "lambda",
            #     "alpha",
            #     "Selected features",
            #     unique_models,
            #     "selected_features",
            #     basename,
            #     x,
            #     if_none_0,
            #     log_scale=True,
            # )

            basename = f"{model_name}_isoline_{opti}_{penal}"
            decorate_multi_and_save(
                fig_isoline,
                model_name,
                lambda_html,
                alpha_html,
                unique_models,
                "selected_features",
                basename,
                x,
                if_none_0,
                log_scale=True,
            )
            if not os.path.isdir("fdr_control_isoline"):
                os.mkdir("fdr_control_isoline")

            basename = f"{model_name}_fdr_control_{opti}_{penal}"
            decorate_multi_and_save(
                fig_isoline_fdr,
                model_name,
                lambda_html,
                alpha_html,
                unique_models,
                "fdr_control_isoline",
                basename,
                x,
                if_none_0,
                log_scale=True,
            )

    # alpha 2 check, alpha 2 depends only on the measure, kernel and data
    alpha2_values = alpha2_table(table)
    alpha2_values.to_csv("alpha2_values.csv")


if __name__ == "__main__":
    main()
