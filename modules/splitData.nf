#!/usr/bin/env nextflow

/*
 * Split Data 
 */
process splitData {

    input:
    path loaded_dataset
    val splitting_strat

    output:
    path "preprocessed_dataset.h5ad"


    script:
    """
    export PYTHONPATH="${projectDir}"
    
    python -m preprocessing.scripts.data_splitting \
        "${splitting_strat}" \
        "${loaded_dataset}" \
        "preprocessed_dataset.h5ad"
    """
}
