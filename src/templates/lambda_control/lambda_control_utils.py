from functools import partial
import numpy as np
from dclasso.dc_lasso import alpha_threshold, loss, minimize_loss
from dclasso import pic_penalty


def fdr(causal_features, selected_features):

    if len(selected_features) == 0:
        print("No feature selection process happened")
        return -1
    else:
        n_selected = len(selected_features)
        intersection = list(set(selected_features) & set(causal_features))
        number_of_correct_positives = len(intersection)
        fdr = (n_selected - number_of_correct_positives) / n_selected
        return fdr


def perform_alpha_computations(
    alpha_list, wjs, screen_indices, causal_features, conservative
):
    run_fdr = []
    run_var_selected = []
    for alpha in alpha_list:

        selected_features, _, _ = alpha_threshold(
            alpha, wjs, screen_indices, conservative=conservative, verbose=False
        )
        selected_features = list(np.array(selected_features))
        run_fdr.append(fdr(causal_features, selected_features))
        run_var_selected.append(selected_features)
    return run_fdr, run_var_selected


def perform_optimisation_with_parameters(
    dclasso_main,
    pen,
    opt,
    lam,
    alpha_list,
    d,
    causal_features,
    key,
    max_epoch,
    eps_stop,
    opt_kwargs,
    Cst,
    penalty_kwargs,
    conservative,
):
    penalty_kwargs = {"name": pen, "lamb": lam}
    loss_fn = partial(
        loss,
        Dxy=dclasso_main.Dxy,
        Dxx=dclasso_main.Dxx,
        penalty_func=pic_penalty(penalty_kwargs),
    )

    step_function, opt_state, beta = dclasso_main.setup_optimisation(
        loss_fn,
        opt,
        d,
        key,
        "from_convex_solve",
        dclasso_main.Dxx,
        dclasso_main.Dxy,
        opt_kwargs,
    )
    beta, _ = minimize_loss(
        step_function,
        opt_state,
        beta,
        max_epoch,
        eps_stop,
        verbose=dclasso_main.verbose,
    )
    wjs = beta[:d] - beta[d:]
    fdr_l, selected_l = perform_alpha_computations(
        alpha_list, wjs, dclasso_main.screen_indices_, causal_features, conservative
    )

    loss_fn = partial(
        loss,
        Dxy=dclasso_main.Dxy,
        Dxx=dclasso_main.Dxx,
        penalty_func=pic_penalty({"name": "None"}),
    )

    loss_train = float(loss_fn(beta))
    loss_fn = partial(
        loss,
        Dxy=dclasso_main.Dxy_val,
        Dxx=dclasso_main.Dxx_val,
        penalty_func=pic_penalty({"name": "None"}),
    )
    loss_valid = float(loss_fn(beta[:d]))
    if pen != "None":
        R = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(beta))
    else:
        R = 0.0
    N1 = np.abs(beta).sum()

    return fdr_l, selected_l, loss_train, loss_valid, R, N1


def build_iterator(penalty_list, optimizer_list, lambdas_list):
    for opt in optimizer_list:
        for pen in penalty_list:
            if pen == "None":
                yield pen, opt, 0
            else:
                for lam in lambdas_list:
                    yield pen, opt, lam
