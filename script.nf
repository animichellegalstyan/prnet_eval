#!/usr/bin/env nextflow

// Include modules
include { loadCompoundMetadata; loadInstanceMetadata; addFingerprints; preprocessMetadata; loadExpressionData } from './modules/preprocessLincs.nf'

/*
 * Pipeline Parameters
 */
params.batch                = "batch_run_00"
params.metadata_folder_path = '/Users/ani/Thesis/prnet_eval/dataset/metadata/LINCS'
params.gctx_cp_path         = '/Users/ani/Thesis/prnet_eval/dataset/data/LINCS/level3_beta_trt_cp_n1805898x12328.gctx'
params.gctx_ctl_path        = '/Users/ani/Thesis/prnet_eval/dataset/data/LINCS/level3_beta_ctl_n188708x12328.gctx' 

workflow {

    main:

    // 1. Initialize our entry-point data channels natively
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

    publish:

    fingerprints_out = addFingerprints.out
    metadata_out     = preprocessMetadata.out
    expression_out   = loadExpressionData.out
}

// 6. Direct the streams to their explicit destination paths on your Mac
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
    

}