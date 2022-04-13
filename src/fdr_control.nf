
nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

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
        tuple path(FILE), path("*.png"), path("*.html")

    script:
        template "fdr_control/plot.py"
}


workflow {
    main:
        simulate_data(params.simulation_models, params.num_samples, params.num_features, repeats)
        fdr_control(simulate_data.out, params.measure_stat, params.kernel, params.penalty, params.optimizer)
        plot(fdr_control.out.collectFile(skip: 1, keepHeader: true))
}
