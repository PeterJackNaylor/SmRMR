

process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"

    input:
        each SIMULATION
        each N_P
        each REPEATS
        val PREFIX
    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation${PREFIX}.npz"), path("causal${PREFIX}.npz")

    script:
        NUM_SAMPLES = N_P[0]
        NUM_FEATURES = N_P[1]
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


workflow simulation {
    main:
        simulate_data(params.simulation_models, params.simulation_np, 1..params.repeat, "")
        validation_data(params.simulation_models, params.validation_samples, 1, "_val")
        test_data(params.simulation_models, [params.test_samples], params.num_features, 1, "_test")

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
