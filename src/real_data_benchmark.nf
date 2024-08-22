CWD = System.getProperty("user.dir")


data_files = files(params.data_location + "/*.npz")

process dclasso {
    time '1d'
    tag "model=DCLasso;data=${TAG});params=(${model_tag};${PENALTY})"
    errorStrategy = 'ignore'
    // maxRetries = 2

    input:
        each path(DATA_NPZ)
        each MS
        each KERNEL
        each PENALTY
        val REPEATS
        val PARAMS_FILE

    output:
        tuple val("data=${TAG}"), path("${METHOD}.csv")

    when:
        (MS == "HSIC") || (KERNEL == "gaussian")

    script:
        TAG = DATA_NPZ.getBaseName()
        if (MS != "HSIC"){
            model_tag = MS
        } else {
            model_tag = "${MS},${KERNEL}"
        }
        METHOD = "DCLasso(${PENALTY},${model_tag})"
        template "feature_selection/DCLasso_realdata.py"
}

process feature_selection {
    time '3h'
    errorStrategy = 'ignore'
    tag "feature_selection=${MODEL.name};data=${TAG})"

    input:
        each MODEL
        each path(DATA_NPZ)
        val REPEATS
        val PARAMS_FILE

    output:
        tuple val("data=${TAG}"), path("${MODEL.name}.csv")

    script:
        TAG = DATA_NPZ.getBaseName()
        template "feature_selection/${MODEL.name}_realdata.py"

}

process groupCSV {
    publishDir "${params.out}/selected_features", mode: 'symlink'
    input:
        tuple val(data), path(CSV)
    output:
        path("${data_name}.csv")
    
    script:
        data_name = "${data}".split("=")[1]
        py = file("src/templates/real_data_group_csv.py")
        """
        python $py $data_name
        
        """



}

config = CWD + "/" + params.config_path

workflow models {
    take:
        data
        feature_selection_methods
        prediction_methods
        dclasso_ms
        dclasso_kernel
        dclasso_penalty
        repeat
        config_file
    main:
        dclasso(data, dclasso_ms, dclasso_kernel, dclasso_penalty, repeat, config_file)
        feature_selection(feature_selection_methods, data, params.repeat, config_file)

        dclasso.out .concat(feature_selection.out) .set{selected_features}
        selected_features.groupTuple().set{csv_tables}

        groupCSV(csv_tables)
        // prediction(prediction_methods, feature_selection_model, config_file)


    // emit:
    //     performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        models(
            data_files, params.feature_selection, params.prediction,
            params.measure_stat, params.kernel, params.penalty, params.repeat, config
        )
        // plot(models.out)
}
