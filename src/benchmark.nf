nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "regression"

config = file("${params.out}/config.yaml")
mode = params.mode

// simulation_models = ['linear_0']
// feature_selection_algorithms = ["dclasso"] //'all_features', "hsic_lasso"
//model_algorithms = ['random_forest'] // 'logistic_regression', 'random_forest', 'svc', 'knn'
// performance_metrics = ['tpr_fpr', 'features_tpr_fpr'] //'auc_roc',



process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"
    afterScript 'mv scores.npz causal.npz'

    input:
        each SIMULATION
        each NUM_SAMPLES
        each NUM_FEATURES
        each REPEATS
    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation.npz"), path('causal.npz')

    when:
        (NUM_SAMPLES < NUM_FEATURES + 1) || ((NUM_SAMPLES == 500) && (NUM_FEATURES == 100))

    script:
        template "simulation/${SIMULATION}.py"

}

process split_data {

    tag "${PARAMS},${I})"

    input:
        tuple val(PARAMS), path(DATA_NPZ), path(CAUSAL_NPZ)
        each I
        val SPLITS

    output:
        tuple val("${PARAMS},${I})"), path("Xy_train.npz"), path("Xy_test.npz"), path(CAUSAL_NPZ)

    script:
        template 'data/kfold.py'

}

process make_grid {

    tag "${PARAMS},${I})"

    input:
        each MODEL
        path PARAMS_FILE

    output:
        tuple val("${PARAMS},${I})"), path("grid.npz")

    script:
        template 'data/grid.py'

}

process feature_selection {

    tag "${MODEL.name};${PARAMS}"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL.name}(${MODEL.parameters});${PARAMS}"), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path('scores_feature_selection.npz')

    script:
        template "feature_selection/${MODEL.name}.py"

}

process prediction {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_model.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ)
        path PARAMS_FILE

    output:
        tuple val("model=${MODEL};${PARAMS}"), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "${mode}/${MODEL}.py"

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


workflow models {
    take: data
    main:
        split_data(data, 0..(params.splits - 1), params.splits)
        feature_selection(params.feature_selection, split_data.out, config)
        prediction(params.prediction, feature_selection.out, config)
        performance(params.performance_metrics, prediction.out)
    emit:
        performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(params.simulation_models, params.num_samples, params.num_features)
        models(simulate_data.out)
}
