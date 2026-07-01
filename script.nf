#!/usr/bin/env nextflow

// Include modules
include { loadCompoundMetadata; loadInstanceMetadata; addFingerprints; preprocessMetadata; loadExpressionData } from './modules/preprocessLincs.nf'
include { splitData } from './modules/splitData.nf'
include { training } from './modules/trainPRnet.nf'

/*
 * Pipeline Parameters
 */
params {

    batch                : String
    splitting_strats      : List<String>
    split_key            : String
    smoke_test           : Boolean 
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

    splitData (
        loadExpressionData.out,
        params.splitting_strats
    )

    /*
    training(
        splitData.out, 
        params.split_key, 
        params.smoke_test
    )*/

    publish:
    fingerprints_out      = addFingerprints.out
    metadata_out          = preprocessMetadata.out
    expression_out        = loadExpressionData.out
    final_dataset_out     = splitData.out.preprocessed_dataset

    //training_loss_out     = training.out.training_loss
    //training_metrics_out  = training.out.training_metrics

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

    /*
    training_loss_out {
        path "${params.batch}/training"
    }

    training_metrics_out {
        path "${params.batch}/training"
    }*/
    
}