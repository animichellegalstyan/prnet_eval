#!/usr/bin/env nextflow

/*
 * Split Data 
 */
process splitData {

    input:
    

    output:

    script:

}

/* - split data in 6:2:2 ratio (i: anndata_df o: train_adata + split_strat, val_adata, test_adata)
 * - split training data into 5 folds, each with 5 blocks train/val/test (i: train_adata, o: train_adata)
 */