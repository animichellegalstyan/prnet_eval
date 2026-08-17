process testing {

    conda '/home/ani/miniconda3/envs/prnet_run_env'

    maxForks 1 // force sequencial run (parallel is default) 

    input:
    path preprocessed_dataset
    tuple val(split_key), path(best_epoch_checkpoint)

    output:
    path "${split_key}_x_true_array.csv", emit: control
    path "${split_key}_y_true_array.csv", emit: ground_truth
    path "${split_key}_y_pre_array.csv", emit: prediction 
    path "${split_key}_cov_drug_array.csv", emit: split_key_file
    path "${split_key}_technical_duplicates.csv", emit: td_splits
    
    script:
    """
    echo "========================================================="
    echo "            RUNNING TESTING LOCALLY                     "
    echo "========================================================="
    echo "localhost" > .node_name

    export PYTHONPATH="${projectDir}"
    python -m test_lincs \
        --input_data ${preprocessed_dataset} \
        --split_key ${split_key} \
        --checkpoint_path ${best_epoch_checkpoint}
    """
}