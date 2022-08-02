nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "regression"

config = params.config_path
mode = params.mode

// simulation_models = ['linear_0']
// feature_selection_algorithms = ["dclasso"] //'all_features', "hsic_lasso"
//model_algorithms = ['random_forest'] // 'logistic_regression', 'random_forest', 'svc', 'knn'
// performance_metrics = ['tpr_fpr', 'features_tpr_fpr'] //'auc_roc',

include { simulate_data; simulate_data as validation_data; simulate_data as test_data} from './nf_core/data_simulation.nf'

process dclasso {
    tag "model=DCLasso;data=${TAG});params=(${PENALTY},${AM},${KERNEL})"
    input:
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        each PENALTY
        each AM
        each KERNEL
        val PARAMS_FILE

    output:
        tuple val("model=DCLasso;data=${TAG});params=(${PENALTY},${AM},${KERNEL})"), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_dclasso.npz"), path('y_proba.npz'), path('y_pred.npz')

    when:
        (AM == "HSIC") || (KERNEL == "linear")

    script:
        template "feature_selection_and_classification/DCLasso_simulations.py"
}

process feature_selection_and_classification {
    tag "model=${MODEL.name};data=${TAG})"
    input:
        each MODEL
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        val PARAMS_FILE

    output:
        tuple val("model=${MODEL.name};data=${TAG})"), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_${MODEL.name}.npz"), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "feature_selection_and_classification/${MODEL.name}.py"
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

process prediction {

    tag "model=${MODEL.name};${PARAMS}"
    afterScript 'mv scores.npz scores_model.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ)
        val PARAMS_FILE

    output:
        tuple val("model=${MODEL.name};${PARAMS}"), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "${mode}/${MODEL.name}.py"

}

process performance {

    tag "${METRIC};${PARAMS}"

    input:
        each METRIC
        tuple val(PARAMS), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path(PROBA_NPZ), path(PRED_NPZ)

    output:
        path 'performance.tsv'

    script:
        template "performance/${METRIC}.py"

}

process plot {

        tag "benchmark"
        publishDir "${params.out}/plots", mode: 'symlink'

        input:
            path PERFORMANCE

        output:
            tuple path(PERFORMANCE), path("*.png"), path("*.html")

        script:
            py = file("src/templates/benchmark/plot.py")
            """
            python $py $PERFORMANCE
            """

}

workflow simulation {
        simulation_models
        num_samples
        validation_samples
        test_samples
        num_features
        repeat
    main:
        simulate_data(simulation_models, num_samples, num_features, 1..repeat, 0, "")
        validation_data(simulation_models, validation_samples, num_features, 1, 0, "_val")
        test_data(simulation_models, test_samples, num_features, 1, 1, "_test")

        simulate_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[0], it[1], it[2]]}
                    .set{ train_split }
        validation_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[1]]}
                    .set{ validation_split }
        test_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[1]]}
                    .set{ test_split }
        train_split .combine(validation_split, by: 0) .combine(test_split, by: 0) .set{ data }
    emit:
        data
}

workflow models {
    take:
        data
        feature_selection_methods
        prediction_methods
        dclasso_penalty
        dclasso_ms
        dclasso_kernel
        feature_and_join_classification_methods
        metrics
    main:
        feature_selection(feature_selection_methods, data, config)
        prediction(prediction_methods, feature_selection.out, config)

        dclasso(data, dclasso_penalty, dclasso_simulations, dclasso_kernel, config)

        feature_selection_and_classification(feature_and_join_classification_methods, data, config)

        dclasso.out .concat(feature_selection_and_classification.out) .concat(prediction.out) .set{ perf }


        performance(metrics, perf)

    emit:
        performance.out.collectFile(name: "${params.out}/performance_${mode}.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulation(
            params.simulation_models, params.num_samples, params.validation_samples,
            params.test_samples, params.num_features, params.repeat
        )
        models(
            simulation.out, params.feature_selection, params.prediction, params.penalty,
            params.measure_stat, params.kernel, params.feature_selection_and_classification,
            params.performance_metrics
        )
        plot(models.out)
}
