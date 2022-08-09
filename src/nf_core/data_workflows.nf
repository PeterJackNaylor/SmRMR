
include { simulate_data; simulate_data as sim_validation} from './data_simulation.nf'


workflow prep_channel_init {
    take:
        simulated_data
    main:
        simulated_data.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[0], it[1], it[2]]}
                            .set{ data_split }
    emit:
        data_split
}

workflow prep_channel_others {
    take:
        simulated_data
    main:
        simulated_data.map{ it -> [[it[0].split('\\(')[0], it[0].split(',')[1]], it[1]]}
                    .set{ data_split }
    emit:
        data_split
}

workflow simulation_train_validation {
    take:
        simulation_models
        train_pairs_np
        validation_pairs_np
        repeat
    main:
        simulate_data(simulation_models, train_pairs_np, 1..repeat,  "")
        prep_channel_init(simulate_data.out).set{ train_split }

        sim_validation(simulation_models, validation_pairs_np, 1, "_val")
        prep_channel_others(sim_validation.out).set{ val_split }

        train_split .combine(val_split, by: 0) .set{ data }
    emit:
        data
}


workflow simulation_train_validation_test {
    take:
        simulation_models
        train_pairs_np
        validation_pairs_np
        test_pairs_np
        repeat
    main:
        simulation_train_validation(simulation_models, train_pairs_np, validation_pairs_np, repeat)

        sim_validation(simulation_models, test_pairs_np, 1, "_test") .set{ sim_test }
        prep_channel_others(sim_test).set{ test_split }

        simulation_train_validation.out .combine(test_split, by: 0) .set{ data }
    emit:
        data
}
