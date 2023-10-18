nextflow.enable.dsl = 2
CWD = System.getProperty("user.dir")
config_file = file(params.config_path)
// Parameters
/////////////////////////////////////////////////////////

include { simulation_train_validation_test_unique } from './nf_core/data_workflows.nf'

process dclasso {
    tag "model=DCLasso);params=(${model_tag};${PENALTY})"
    errorStrategy = 'retry'
    maxRetries = 2

    input:
        path ALL
        each MS
        each KERNEL
        each PENALTY
        val PARAMS_FILE

    output:
        path("results*.csv")

    when:
        (MS == "HSIC") || (KERNEL == "linear")

    script:
        pyfile = file("src/templates/feature_selection/DCLasso_simulations_lessjobs.py")
        if (MS != "HSIC"){
            model_tag = MS
        } else {
            model_tag = "${MS},${KERNEL}"
        }
        """
        python $pyfile $MS $KERNEL $PENALTY $PARAMS_FILE
        """
}


process feature_selection {

    tag "feature_selection=${MODEL.name})"

    input:
        each MODEL
        path ALL
        val PARAMS_FILE

    output:
        path("results*.csv")

    script:
        pyfile = file("src/templates/feature_selection/${MODEL.name}_lessjobs.py")
        """
        python $pyfile $PARAMS_FILE
        """

}


process table_and_plot {

    tag "Plot/Table"

    input:
        path ALL

    output:
        path("all_results.csv")

    script:
        """
        #!/usr/bin/env python

        from glob import glob
        import pandas as pd
        files = glob('*.csv')
        dataframes = [pd.read_csv(f) for f in files]
        pd.concat(dataframes, axis=0).to_csv('all_results.csv')
        """

}
workflow models {
    take:
        data
        feature_selection_methods
        prediction_methods
        dclasso_ms
        dclasso_kernel
        dclasso_penalty
        metrics
        config_file
    main:
        data.flatten().filter(~/.*?\.npz/).set{ all_npz }
        dclasso(all_npz.collect(), dclasso_ms, dclasso_kernel, dclasso_penalty, config_file)
        feature_selection(feature_selection_methods, all_npz.collect(), config_file)
        dclasso.out.concat(feature_selection.out).set{results}
        table_and_plot(results.collect())
    //     dclasso.out .concat(feature_selection.out) .set{feature_selection_model}
    //     prediction(prediction_methods, feature_selection_model, config_file)

    //     performance(metrics, prediction.out)

    // emit:
    //     performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulation_train_validation_test_unique(
            params.simulation_models, params.simulation_np, params.validation_samples,
            params.test_samples, params.repeat
        )
        models(
            simulation_train_validation_test_unique.out, params.feature_selection, params.prediction,
            params.measure_stat, params.kernel, params.penalty, params.performance_metrics,
            config_file
        )
}
