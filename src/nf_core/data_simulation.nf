

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


process simulate_data_unique {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"
    afterScript "mv simulation${PREFIX}.npz simulation${PREFIX}__${SIMULATION}-${NUM_SAMPLES}-${NUM_FEATURES}__${REPEATS}.npz ; mv causal${PREFIX}.npz causal${PREFIX}__${SIMULATION}-${NUM_SAMPLES}-${NUM_FEATURES}__${REPEATS}.npz"
    input:
        each SIMULATION
        each N_P
        each REPEATS
        val PREFIX
    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation*.npz"), path("causal*.npz")

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
