nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "regression"

config = file("${params.out}/config.yaml")
mode = params.mode

num_samples = [400]
num_features = [500]

simulation_models = ['linear_0']
feature_selection_algorithms = ["dclasso"] //'all_features', "hsic_lasso"
model_algorithms = ['random_forest'] // 'logistic_regression', 'random_forest', 'svc', 'knn'
performance_metrics = ['tpr_fpr', 'features_tpr_fpr'] //'auc_roc',

process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"
    afterScript 'mv scores.npz causal.npz'

    input:
        each SIMULATION
        each NUM_SAMPLES
        each NUM_FEATURES

    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation.npz"), path('causal.npz')

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

process feature_selection {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL};${PARAMS}"), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path('scores_feature_selection.npz')

    script:
        template "feature_selection/${MODEL}.py"

}

process model {

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
        feature_selection(feature_selection_algorithms, split_data.out, config)
        model(model_algorithms, feature_selection.out, config)
        performance(performance_metrics, model.out)
    emit:
        performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(simulation_models, num_samples, num_features)
        models(simulate_data.out)
}
