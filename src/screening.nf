
nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulate_data } from './nf_core/data_simulation.nf'

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
        tuple path(FILE), path("*.png"), path("*.html")

    script:
        template "screening/plot.py"
}


workflow {
    main:

        simulate_data(params.simulation_models, train_pairs_np, repeats, "")
        screen(simulate_data.out, params.measure_stat, params.kernel)
        plot(screen.out.collectFile(skip: 1, keepHeader: true))
}
