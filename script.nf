#!/usr/bin/env nextflow

// Include modules
include { loadCompoundMetadata; loadInstanceMetadata; addFingerprints; preprocessMetadata; loadExpressionData; normalizeAndSplit } from './modules/preprocessLincs.nf'

/*
 * Pipeline Parameters
 */
params {

    batch                : String
    metadata_folder_path : Path
    gctx_cp_path         : Path
    gctx_ctl_path        : Path

}

workflow {

    main:

    // 1. Initialize entry-point data channels 
    comp_file_ch   = channel.fromPath("${params.metadata_folder_path}/compoundinfo_beta.txt")
    inst_file_ch   = channel.fromPath("${params.metadata_folder_path}/instinfo_beta.txt")
    gene_file_ch   = channel.fromPath("${params.metadata_folder_path}/geneinfo_beta.txt")
    gctx_cp_ch     = channel.fromPath(params.gctx_cp_path)
    gctx_ctl_ch    = channel.fromPath(params.gctx_ctl_path)
    
    loadCompoundMetadata(comp_file_ch)
    loadInstanceMetadata(inst_file_ch)
    
    addFingerprints(loadCompoundMetadata.out)
    preprocessMetadata(loadInstanceMetadata.out)

    loadExpressionData(
        addFingerprints.out,
        preprocessMetadata.out,
        gene_file_ch, 
        gctx_cp_ch,
        gctx_ctl_ch
    )

    normalizeAndSplit(
        loadExpressionData.out,
        params.splitting_strat
    )

    publish:
    fingerprints_out  = addFingerprints.out
    metadata_out      = preprocessMetadata.out
    expression_out    = loadExpressionData.out
    final_dataset_out = normalizeAndSplit.out
}

output {
    fingerprints_out {
        path "${params.batch}/intermediates"
    }
    metadata_out {
        path "${params.batch}/intermediates"
    }

    expression_out {
       path "${params.batch}/intermediates"
    }
    
    final_dataset_out { 
        path "${params.batch}/intermediates" 
    }
    

}