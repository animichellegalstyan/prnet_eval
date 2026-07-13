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
    splitting_strats     : List<String>
    train_split          : String
    smoke_test           : Boolean 
    metadata_folder_path : Path
    gctx_cp_path         : Path
    gctx_ctl_path        : Path

}

workflow {

    main:

    // 1. Initialize data channels 
    comp_file_ch        = channel.fromPath("${params.metadata_folder_path}/compoundinfo_beta.txt")
    inst_file_ch        = channel.fromPath("${params.metadata_folder_path}/instinfo_beta.txt")
    gene_file_ch        = channel.fromPath("${params.metadata_folder_path}/geneinfo_beta.txt")
    gene_expr_cp_ch     = channel.fromPath(params.gctx_cp_path)
    gene_expr_ctl_ch    = channel.fromPath(params.gctx_ctl_path)

    // Initialize fold-channels
    def split_key_ch = Channel.of(0..4).map { fold -> "${params.train_split}_split_${fold}" }

    loadCompoundMetadata(comp_file_ch)
    loadInstanceMetadata(inst_file_ch)
    
    addFingerprints(loadCompoundMetadata.out)
    preprocessMetadata(loadInstanceMetadata.out)

    loadExpressionData(
        addFingerprints.out,
        preprocessMetadata.out,
        gene_file_ch, 
        gene_expr_cp_ch,
        gene_expr_ctl_ch
    )

    splitData (
        loadExpressionData.out,
        params.splitting_strats
    )
 
    training (
        splitData.out.preprocessed_dataset, 
        split_key_ch, 
        params.smoke_test
    )

    publish:
    fingerprints_out      = addFingerprints.out
    metadata_out          = preprocessMetadata.out
    expression_out        = loadExpressionData.out
    final_dataset_out     = splitData.out.preprocessed_dataset

    training_loss_out     = training.out.training_loss
    training_metrics_out  = training.out.training_metrics

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

    training_loss_out {
        path "${params.batch}/training_${params.train_split}"
    }

    training_metrics_out {
        path "${params.batch}/training_${params.train_split}"
    }
    
}