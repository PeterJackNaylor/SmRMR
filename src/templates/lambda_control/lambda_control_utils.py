from functools import partial
import numpy as np
from smrmr.smrmr_class import loss
from smrmr.utils import alpha_threshold, minimize_loss
from smrmr import pic_penalty


def fdr(causal_features, selected_features, verbose=False):

    if not len(selected_features):
        if verbose:
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
    smrmr_main,
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
    beta, value, warm_start = smrmr_main.cvx_solve(penalty_kwargs)
    if warm_start:
        loss_fn = smrmr_main.compute_loss_fn(penalty_kwargs)

        step_function, opt_state, beta = smrmr_main.setup_optimisation(
            loss_fn,
            opt,
            beta,
            opt_kwargs,
        )
        beta, _ = minimize_loss(
            step_function,
            opt_state,
            beta,
            max_epoch,
            eps_stop,
            verbose=smrmr_main.verbose,
        )
    wjs = beta[:d] - beta[d:]
    fdr_l, selected_l = perform_alpha_computations(
        alpha_list, wjs, smrmr_main.screen_indices_, causal_features, conservative
    )

    loss_fn = partial(
        loss,
        Dxy=smrmr_main.Dxy,
        Dxx=smrmr_main.Dxx,
        penalty_func=pic_penalty({"name": "None"}),
    )

    loss_train = float(loss_fn(beta))
    loss_fn = partial(
        loss,
        Dxy=smrmr_main.Dxy_val,
        Dxx=smrmr_main.Dxx_val,
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


def build_ms_kern_iterator(ms_list, kernel_list):
    for ms in ms_list:
        if ms == "HSIC":
            for kernel in kernel_list:
                yield ms, kernel
        else:
            yield ms, ""


def length_ms_kern_iterator(ms_list, kernel_list):
    n = 0
    if "HSIC" in ms_list:
        n += len(kernel_list)
        n += len(ms_list) - 1
    else:
        n += len(ms_list)
    return n


def length_iterator(penalty_list, optimizer_list, lambdas_list):
    n_opt = len(optimizer_list)
    if "None" in penalty_list:
        n_pen_lamb = (len(penalty_list) - 1) * len(lambdas_list) + 1
    else:
        n_pen_lamb = len(penalty_list) * len(lambdas_list)
    return n_opt * n_pen_lamb
