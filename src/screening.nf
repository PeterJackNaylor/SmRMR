
nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'

num_samples = [100, 500]
num_features = [100, 500]
repeats = 0..199

kernel = ["linear", "gaussian"]
measure_stat = ["DC", "HSIC"]

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = kernel[0]

simulation_models = ['categorical_1', 'categorical_2']

include { simulate_data } from './benchmark.nf'

process screen {
    tag "${PARAMS},${AM},${KERNEL})"
    input:
        tuple val(PARAMS), path(DATA_NPZ), path(CAUSAL_NPZ)
        each AM
        each KERNEL
    output:
        path "performance.tsv"

    when:
        ((AM in KERNEL_AM) || (KERNEL == FIRST_KERNEL))

    script:
        template "screening/main.py"
}

process plot {
    publishDir "${params.out}", mode: 'symlink'
    input:
        path FILE

    output:
        tuple path(FILE), path("*.png")

    script:
        template "screening/plot.py"
}


workflow {
    main:
        simulate_data(simulation_models, num_samples, num_features, repeats)
        screen(simulate_data.out, measure_stat, kernel)
        plot(screen.out.collectFile(skip: 1, keepHeader: true))
}
