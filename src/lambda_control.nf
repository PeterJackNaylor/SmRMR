nextflow.enable.dsl = 2
CWD = System.getProperty("user.dir")
// Parameters
/////////////////////////////////////////////////////////
repeats = 0..(params.repeat-1)

KERNEL_AM = ["HSIC"]
FIRST_KERNEL = params.kernel[0]

include { simulation_train_validation } from './nf_core/data_workflows.nf'


process lambda_control {

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
