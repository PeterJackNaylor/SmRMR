nextflow.enable.dsl = 2
CWD = System.getProperty("user.dir")

// Parameters
/////////////////////////////////////////////////////////

include { simulation_train_validation_test } from './nf_core/data_workflows.nf'

process dclasso {
    tag "model=DCLasso;data=${TAG});params=(${PENALTY})"
    input:
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        each PENALTY
        val PARAMS_FILE

    output:
        tuple val("feature_selection=DCLasso(${PENALTY});data=${TAG})"), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_dclasso.npz")

    script:
        template "feature_selection/DCLasso_simulations.py"
}

process feature_selection {

    tag "feature_selection=${MODEL.name};data=${TAG})"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        each MODEL
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        val PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL.name};data=${TAG})"), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_feature_selection_${MODEL.name}.npz")

    script:
        template "feature_selection/${MODEL.name}.py"

}

// process feature_selection_and_prediction {
//     tag "model=${MODEL.name};data=${TAG})"
//     input:
//         each MODEL
//         tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
//         val PARAMS_FILE

//     output:
//         tuple val("model=${MODEL.name};feature_selection=${MODEL.name};data=${TAG})"), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_${MODEL.name}.npz"), path('y_proba.npz'), path('y_pred.npz')

//     script:
//         template "feature_selection_and_prediction/${MODEL.name}.py"
// }

process prediction {

    tag "model=${MODEL.name};${PARAMS}"
    afterScript 'mv new_scores.npz prediction_scores.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ)
        val PARAMS_FILE

    output:
        tuple val("model=${MODEL.name};${PARAMS}"), path(TEST_NPZ), path(CAUSAL_NPZ), path("prediction_scores.npz"), path('y_proba.npz'), path('y_pred.npz')

    when:
        ("${PARAMS}".contains("linear") & "${MODEL.prediction}" == "regression") || (!("${PARAMS}".contains("linear")) & "${MODEL.prediction}" == "classification")

    script:
        template "${MODEL.prediction}/${MODEL.name}.py"

}

process performance {

    tag "${METRIC.name};${PARAMS}"

    input:
        each METRIC
        tuple val(PARAMS), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path(PROBA_NPZ), path(PRED_NPZ)

    output:
        path 'performance.tsv'

    when:
        ("${PARAMS}".contains("linear") & "${METRIC.prediction}" == "regression") || (!("${PARAMS}".contains("linear")) & "${METRIC.prediction}" == "classification") || ("${METRIC.prediction}" == "both")

    script:
        template "performance/${METRIC.name}.py"

}

process plot {

        tag "benchmark"
        publishDir "${params.out}/plots", mode: 'symlink'

        input:
            path PERFORMANCE

        output:
            tuple path(PERFORMANCE), path("*.png"), path("*.html")

        script:
            py = file("src/templates/benchmark/line_plot.py")
            """
            python $py $PERFORMANCE
            """

}

config = CWD + "/" + params.config_path

workflow models {
    take:
        data
        feature_selection_methods
        prediction_methods
        dclasso_penalty
        metrics
        config_file
    main:
        dclasso(data, dclasso_penalty, config_file)
        feature_selection(feature_selection_methods, data, config_file)

        dclasso.out .concat(feature_selection.out) .set{feature_selection_model}
        prediction(prediction_methods, feature_selection_model, config_file)

        performance(metrics, prediction.out)

    emit:
        performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulation_train_validation_test(
            params.simulation_models, params.simulation_np, params.validation_samples,
            params.test_samples, params.repeat
        )
        models(
            simulation_train_validation_test.out, params.feature_selection, params.prediction,
            params.penalty, params.performance_metrics,
            config
        )
        plot(models.out)
}
