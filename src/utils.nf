

process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"

    input:
        each SIMULATION
        each NUM_SAMPLES
        each NUM_FEATURES
        each REPEATS
        val FORCE
        val NAME
    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation${NAME}.npz"), path("causal${NAME}.npz")

    when:
        ((NUM_SAMPLES < NUM_FEATURES + 1) || ((NUM_SAMPLES == 500) && (NUM_FEATURES == 100))) || (FORCE == 1)

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
