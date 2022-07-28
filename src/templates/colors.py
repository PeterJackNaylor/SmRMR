color_dictionnary = {
    "cMMD": "rgb(239,59,44)",
    "HSIC": "rgb(107,174,214)",
    "HSIC_norm": "rgb(0,109,44)",
    "TR": "rgb(253,191,111)",
    "pearson_correlation": "rgb(106,61,154)",
    "PC": "rgb(116,196,118)",
    "DC": "rgb(177,89,40)",
}

precise_color_dictionnary = {
    "cMMD(linear)": "rgb(165,15,21)",
    "cMMD(distance)": "rgb(252,187,161)",
    "cMMD(gaussian)": "rgb(239,59,44)",
    "HSIC(linear)": "rgb(222,235,247)",
    "HSIC(distance)": "rgb(198,219,239)",
    "HSIC(sigmoid)": "rgb(158,202,225)",
    "HSIC(inverse-M)": "rgb(107,174,214)",
    "HSIC(tanh)": "rgb(66,146,198)",
    "HSIC(laplacian)": "rgb(33,113,181)",
    "HSIC(gaussian)": "rgb(8,81,156)",
    "HSIC_norm": "rgb(116,196,118)",
    "TR": "rgb(253,191,111)",
    "Pearson": "rgb(106,61,154)",
    "PC": "rgb(116,196,118)",
    "DC": "rgb(177,89,40)",
}

kernel_colours = {
    "cMMD": {
        "linear": "rgb(165,15,21)",
        "gaussian": "rgb(251,106,74)",
        "distance": "rgb(252,187,161)",
    },
    "HSIC": {
        "linear": "rgb(222,235,247)",
        "distance": "rgb(198,219,239)",
        "sigmoid": "rgb(158,202,225)",
        "inverse-M": "rgb(107,174,214)",
        "tanh": "rgb(66,146,198)",
        "laplacian": "rgb(33,113,181)",
        "gaussian": "rgb(8,81,156)",
    },
    "HSIC_norm": {
        "linear": "rgb(0,109,44)",
        "gaussian": "rgb(116,196,118)",
        "distance": "rgb(199,233,192)",
    },
}

mapping_data_name = {
    "categorical_1": "Cat.1",
    "categorical_2": "Cat.2",
    "categorical_3": "Cat.3",
    "linear_0": "Lin.0",
    "linear_1": "Lin.1",
    "linear_2": "Lin.2",
    "linear_3": "Lin.3",
    "nonlinear_1": "NLin.1",
    "nonlinear_2": "NLin.2",
    "nonlinear_3": "NLin.3",
    "nonlinear_4": "NLin.4",
    "nonlinear_5": "NLin.5",
    "model_1a": "1.a",
    "model_1b": "1.b",
    "model_1c": "1.c",
    "model_2a": "2.a",
    "model_2b": "2.b",
    "model_2c": "2.c",
    "model_2d": "2.d",
    "model_2e": "2.e",
    "model_3a": "3.a",
    "model_3b": "3.b",
    "model_3c": "3.c",
}


def color_dictionary_fdr(name, kernel):
    if kernel != "unspecified":
        return kernel_colours[name][kernel]
    else:
        return color_dictionnary[name]


inside_colors = {
    "unspecified": "rgb(247,247,247)",
    "linear": "rgb(247,247,247)",
    "distance": "rgb(217,217,217)",
    "sigmoid": "rgb(189,189,189)",
    "inverse-M": "rgb(150,150,150)",
    "tanh": "rgb(115,115,115)",
    "laplacian": "rgb(82,82,82)",
    "gaussian": "rgb(37,37,37)",
}


def name_mapping(name, kernel):
    if name == "pearson_correlation":
        return "Pearson"
    else:
        return name


def name_mapping_fdr(name, kernel):
    if name in ["HSIC", "cMMD"]:
        return name + f" ({kernel})"
    elif name == "pearson_correlation":
        return "Pearson"
    else:
        return name


def helper(name, kernel):
    # return name_mapping_fdr(name.split("_")[0], kernel)
    return name_mapping_fdr(name, kernel)


color_dictionary_fdr_keys = [
    helper(am, kernel)
    for am in color_dictionnary.keys()
    for kernel in inside_colors.keys()
]
