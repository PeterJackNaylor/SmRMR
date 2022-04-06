
nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'

num_samples = [100, 500]
num_features = [100, 500]
repeats = 0..199

kernel = ["linear", "gaussian"]
measure_stat = ["DC", "HSIC"]
penalty = ["none", "l1"]//, "scad", "mcp"]
optimizer = ["SGD"]//, "adam"]


KERNEL_AM = ["HSIC"]
FIRST_KERNEL = kernel[0]
simulation_models = ['categorical_1', 'categorical_2']

include { simulate_data } from './benchmark.nf'

process fdr_control {
    tag "${PARAMS},${AM},${KERNEL})"
    input:
        tuple val(PARAMS), path(DATA_NPZ), path(CAUSAL_NPZ)
        each AM
        each KERNEL
        each PENALTY
        each OPTIMIZER
    output:
        path "performance.tsv"

    when:
        ((AM in KERNEL_AM) || (KERNEL == FIRST_KERNEL))

    script:
        template "fdr_control/main.py"
}

process plot {
    publishDir "${params.out}", mode: 'symlink'
    input:
        path FILE

    output:
        tuple path(FILE), path("*.png")

    script:
        template "fdr_control/plot.py"
}


workflow {
    main:
        simulate_data(simulation_models, num_samples, num_features, repeats)
        fdr_control(simulate_data.out, measure_stat, kernel, penalty, optimizer)
        plot(fdr_control.out.collectFile(skip: 1, keepHeader: true))
}
