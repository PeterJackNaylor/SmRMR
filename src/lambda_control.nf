nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulate_data; simulate_data as validation_data} from './nf_core/data_simulation.nf'


process lambda_control {

    tag "${PARAMS},${AM},${KERNEL}"
    input:
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ)
        each AM
        each KERNEL
        each PENALTY
        each OPTIMIZER
        each LAMBDA
    output:
        path "performance.tsv"

    when:
        ((AM in KERNEL_AM) || (KERNEL == FIRST_KERNEL)) && ((LAMBDA == FIRST_LAMBDA) || (PENALTY != "none"))

    script:
        template "lambda_control/main.py"
}


process plot {
    publishDir "${params.out}", mode: 'symlink', overwrite: 'true'

    input:
        path FILE

    output:
        tuple path(FILE), path("loss_train"), path("loss_validation"), path("alpha_fdr"), path("selected_features")

    script:
        template "lambda_control/plot.py"
}


workflow simulation {
    take:
        simulation_models
        num_samples
        validation_samples
        num_features
        repeat
    main:
        simulate_data(simulation_models, num_samples, num_features, 1..repeat, 0, "")
        validation_data(simulation_models, validation_samples, num_features, 1, 0, "_val")

        simulate_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[0], it[1], it[2]]}
                    .set{ train_split }
        validation_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[1]]}
                    .set{ validation_split }
        train_split .combine(validation_split, by: 0) .set{ data }
    emit:
        data
}


workflow {
    main:
        simulation(
            params.simulation_models, params.num_samples, params.validation_samples,
            params.num_features, params.repeat
        )

        lambda_control(simulation.out, params.measure_stat, params.kernel, params.penalty, params.optimizer, params.lambda)
        plot(lambda_control.out.collectFile(skip: 1, keepHeader: true))
}
