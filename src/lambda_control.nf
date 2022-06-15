nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulate_data } from './utils.nf'
include { fdr_control } from './fdr_control.nf'



process plot {
    publishDir "${params.out}", mode: 'symlink'
    input:
        path FILE

    output:
        tuple path(FILE), path("*.png"), path("*.html")

    script:
        template "lambda_control/plot.py"
}


workflow {
    main:
        simulate_data(params.simulation_models, params.num_samples, params.num_features, repeats)
        fdr_control(simulate_data.out, params.measure_stat, params.kernel, params.penalty, params.optimizer, params.lambda)
        plot(fdr_control.out.collectFile(skip: 1, keepHeader: true))
}
