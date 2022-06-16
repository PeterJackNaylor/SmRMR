nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "regression"

config = file(params.config_path)
mode = params.mode

// simulation_models = ['linear_0']
// feature_selection_algorithms = ["dclasso"] //'all_features', "hsic_lasso"
//model_algorithms = ['random_forest'] // 'logistic_regression', 'random_forest', 'svc', 'knn'
// performance_metrics = ['tpr_fpr', 'features_tpr_fpr'] //'auc_roc',

include { simulate_data; simulate_data as validation_data} from './utils.nf'
FORCE = 1
include { simulate_data as test_data } from './utils.nf'

process dclasso {
    tag "model=DCLasso;${TAG});${PENALTY};${AM};${KERNEL}"
    conda { AM == 'PC' ? '/data/ghighdim/pnaylor/project/dclasso/env_GPU/' : '/data/ghighdim/pnaylor/project/dclasso/env/'}
    clusterOptions { task.attempt <= 1 ? (AM == 'PC' ? '-jc gpu-container_g1 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH -ac d=nvcr-cuda-11.1-cudnn8.0': '-v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH') : (AM == 'PC' ? '-jc gs-container_g1 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH -ac d=nvcr-cuda-11.1-cudnn8.0' : '-jc pcc-large -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH') }
    errorStrategy 'retry'
    maxRetries 2
    input:
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        path PARAMS_FILE
        each PENALTY
        each AM
        each KERNEL

    output:
        tuple val("model=DCLasso;${TAG};${PENALTY};${AM};${KERNEL})"), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_dclasso.npz"), path('y_proba.npz'), path('y_pred.npz')

    when:
        (AM == "HSIC") || (KERNEL == "linear")

    script:
        template "feature_selection_and_classification/DCLasso_simulations.py"
}

process feature_selection_and_classification {
    tag "model=${MODEL};${TAG})"
    input:
        each MODEL
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        path PARAMS_FILE

    output:
        tuple val("model=${MODEL};${TAG}"), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_${MODEL.name}.npz"), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "feature_selection_and_classification/${MODEL.name}.py"
}

process feature_selection {

    tag "${MODEL.name};${PARAMS}"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        each MODEL
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ), path(TEST_NPZ)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL.name}(${MODEL.parameters});${PARAMS}"), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path("scores_feature_selection_${MODEL.name}.npz")

    script:
        template "feature_selection/${MODEL.name}.py"

}

process prediction {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_model.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(VAL_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ)
        path PARAMS_FILE

    output:
        tuple val("model=${MODEL};${PARAMS}"), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path('y_proba.npz'), path('y_pred.npz')

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

workflow simulation {
    main:
        simulate_data(params.simulation_models, params.num_samples, params.num_features, 1..params.repeat, 0, "")
        validation_data(params.simulation_models, params.validation_samples, params.num_features, 1, 0, "_val")
        test_data(params.simulation_models, params.test_samples, params.num_features, 1, 1, "_test")

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
    take: data
    main:
        feature_selection(params.feature_selection, data, config)
        prediction(params.prediction, feature_selection.out, config)

        dclasso(data, config, params.penalty, params.measure_stat, params.kernel)

        feature_selection_and_classification(params.feature_selection_and_classification, data, config)

        dclasso.out .concat(feature_selection_and_classification.out) .concat(prediction.out) .set{ perf }


        performance(params.performance_metrics, perf)

    emit:
        performance.out.collectFile(name: "${params.out}/performance_${mode}.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulation()
        models(simulation.out)
}
