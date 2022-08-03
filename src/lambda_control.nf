nextflow.enable.dsl = 2
CWD = System.getProperty("user.dir")
// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulate_data; simulate_data as sim_validation} from './nf_core/data_simulation.nf'


process lambda_control {
    conda { AM == 'PC' ? '/data/ghighdim/pnaylor/project/dclasso/env_GPU/' : '/data/ghighdim/pnaylor/project/dclasso/env/'}
    clusterOptions { task.attempt <= 1 ? (AM == 'PC' ? '-jc gpu-container_g1 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH -ac d=nvcr-cuda-11.1-cudnn8.0': '-v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH') : (AM == 'PC' ? '-jc gs-container_g1 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH -ac d=nvcr-cuda-11.1-cudnn8.0' : 'fail') }
    beforeScript { AM = 'PC' ? 'export LD_LIBRARY_PATH=/data/ghighdim/pnaylor/project/dclasso/env_GPU/lib/:$LD_LIBRARY_PATH' : ''}
    tag "${TAG}),${AM},${KERNEL}"
    input:
        tuple val(PARAMS), val(TAG), path(TRAIN_NPZ), path(CAUSAL_NPZ), path(VAL_NPZ)
        each AM
        each KERNEL
        val PARAMS_FILE
    output:
        path "performance.tsv"

    when:
        ((AM in KERNEL_AM) || (KERNEL == FIRST_KERNEL))

    script:
        template "lambda_control/main.py"
}


process plot {
    publishDir "${params.out}", mode: 'symlink', overwrite: 'true'

    input:
        path FILE

    output:
        tuple path(FILE), path("loss_train"), path("loss_validation"), path("alpha_fdr"), path("fdr_control_isoline"), path("selected_features"), path("R_constraint")

    script:
        template "lambda_control/plot.py"
}


config = CWD + "/" + params.config_path


workflow simulation {
    take:
        simulation_models
        train_pairs_np
        validation_pairs_np
        repeat
    main:
        simulate_data(simulation_models, train_pairs_np, 1..repeat,  "")
        sim_validation(simulation_models, validation_pairs_np, 1, "_val")

        simulate_data.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[0], it[1], it[2]]}
                    .set{ train_split }
        sim_validation.out.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[1]]}
                    .set{ validation_split }
        train_split .combine(validation_split, by: 0) .set{ data }
    emit:
        data
}


workflow {
    main:
        simulation(
            params.simulation_models, params.simulation_np,
            params.validation_samples, params.repeat
        )

        lambda_control(simulation.out, params.measure_stat, params.kernel, config)
        plot(lambda_control.out.collectFile(skip: 1, keepHeader: true))
}
