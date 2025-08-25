CWD = System.getProperty("user.dir")


data_files = files(params.data_location + "/*.npz")

process smrmr {
    time '1d'
    tag "model=smrmr;data=${TAG});params=(${model_tag};${PENALTY})"
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
        tuple val("data=${TAG}"), path("${METHOD}.csv"), path("*_scores.txt")

    when:
        (MS == "HSIC") || (KERNEL == "gaussian")

    script:
        TAG = DATA_NPZ.getBaseName()
        if (MS != "HSIC"){
            model_tag = MS
        } else {
            model_tag = "${MS},${KERNEL}"
        }
        METHOD = "smrmr(${PENALTY},${model_tag})"
        template "feature_selection/smrmr_realdata.py"
}

process feature_selection {
    time '24h'
    errorStrategy = 'ignore'
    tag "feature_selection=${MODEL.name};data=${TAG})"

    input:
        each MODEL
        each path(DATA_NPZ)
        val REPEATS
        val PARAMS_FILE

    output:
        tuple val("data=${TAG}"), path("${MODEL.name}.csv"), path("*_scores.txt")

    script:
        TAG = DATA_NPZ.getBaseName()
        template "feature_selection/${MODEL.name}_realdata.py"

}

process groupCSVandTXT {
    publishDir "${params.out}/selected_features", mode: 'symlink'
    input:
        tuple val(data), path(CSV), path(TXT)
    output:
        tuple path("${data_name}.csv"), path("${data_name}_scores.csv")
    
    script:
        data_name = "${data}".split("=")[1]
        py = file("src/templates/real_data_group_csv_and_txt.py")
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
        smrmr_ms
        smrmr_kernel
        smrmr_penalty
        repeat
        config_file
    main:
        smrmr(data, smrmr_ms, smrmr_kernel, smrmr_penalty, repeat, config_file)
        feature_selection(feature_selection_methods, data, params.repeat, config_file)

        smrmr.out .concat(feature_selection.out) .set{selected_features}
        selected_features.groupTuple().set{csv_tables}

        groupCSVandTXT(csv_tables)
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
