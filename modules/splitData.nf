#!/usr/bin/env nextflow

/*
 * Split Data 
 */
process splitData {
    
    input:
    path loaded_dataset
    val splitting_strats

    output:
    path "preprocessed_dataset.h5ad", emit: preprocessed_dataset
    path "data_split_distributions.log", emit: split_log


    script:
    //handling list input 
    def strats_list = splitting_strats instanceof List ? splitting_strats.join(',') : splitting_strats
    """
    export PYTHONPATH="${projectDir}"
    
    python -m preprocessing.scripts.data_splitting \
        "${strats_list}" \
        "${loaded_dataset}" \
        "preprocessed_dataset.h5ad" \
        > data_split_distributions.log 2>&1
    """
}
