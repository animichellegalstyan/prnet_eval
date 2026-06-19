#!/usr/bin/env nextflow

/*
 * Split Data 
 */
process splitData {

    input:
    path loaded_dataset
    val splitting_strats

    output:
    path "preprocessed_dataset.h5ad"


    script:
    //handling list input 
    def strats_list = splitting_strats instanceof List ? splitting_strats.join(',') : splitting_strats
    """
    export PYTHONPATH="${projectDir}"
    
    python -m preprocessing.scripts.data_splitting \
        "${strats_list}" \
        "${loaded_dataset}" \
        "preprocessed_dataset.h5ad"
    """
}
