nextflow.enable.dsl = 2
CWD = System.getProperty("user.dir")
// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulation_train_validation } from './nf_core/data_workflows.nf'


process lambda_control {
    conda { AM == 'PC' ? '/data/ghighdim/pnaylor/project/dclasso/env_GPU/' : '/data/ghighdim/pnaylor/project/dclasso/env/'}
    clusterOptions { AM == 'PC' ? '-jc gpu-container_g1 -v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH -ac d=nvcr-cuda-11.1-cudnn8.0': '-v PATH=/usr/bin:/home/pnaylor/miniconda3/bin:$PATH,PYTHONPATH=/data/ghighdim/pnaylor/project/dclasso/:/data/ghighdim/pnaylor/project/dclasso/src/templates:$PYTHONPATH' }
    tag "${TAG}),${AM},${KERNEL}"
    beforeScript { AM == 'PC' ? 'export LD_LIBRARY_PATH=/data/ghighdim/pnaylor/project/dclasso/env_GPU/lib:$LD_LIBRARY_PATH' : 'export LD_LIBRARY_PATH=/data/ghighdim/pnaylor/project/dclasso/env/lib:$LD_LIBRARY_PATH'}
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
        tuple path(FILE), path("fdr_alpha.*"), path("n_selected.*"), path("true_features.*")

    script:
        py = file("src/templates/lambda_control/line_plot.py")
        """
        python $py
        """
}


config = CWD + "/" + params.config_path


workflow {
    main:
        simulation_train_validation(
            params.simulation_models, params.simulation_np,
            params.validation_samples, params.repeat
        )

        lambda_control(simulation_train_validation.out, params.measure_stat, params.kernel, config)
        plot(lambda_control.out.collectFile(skip: 1, keepHeader: true))
}
