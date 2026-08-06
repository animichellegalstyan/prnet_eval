#!/usr/bin/env nextflow

// Include modules
include { loadCompoundMetadata; loadInstanceMetadata; addFingerprints; updateAnnDataFingerprints; preprocessMetadata; loadExpressionData; delFalseDuplicates } from './modules/preprocessLincs.nf'
include { splitData } from './modules/splitData.nf'
include { training } from './modules/trainPRnet.nf'
include { testing } from './modules/testPRnet.nf'

/*
 * Pipeline Parameters
 */
params {

    batch                : String
    splitting_strats     : List<String>
    train_split          : String
    smoke_test           : Boolean 
    delete_fp_duplicates : Boolean
    embedding            : String

    metadata_folder_path : Path
    gctx_cp_path         : Path
    gctx_ctl_path        : Path
    split_dataset_path   : Path 

}

workflow {

    main:

        // Preparing data 
        if (params.split_dataset_path) {

            split_data_ch = Channel.fromPath(params.split_dataset_path)

        } else {

            comp_file_ch        = Channel.fromPath("${params.metadata_folder_path}/compoundinfo_beta.txt")
            inst_file_ch        = Channel.fromPath("${params.metadata_folder_path}/instinfo_beta.txt")
            gene_file_ch        = Channel.fromPath("${params.metadata_folder_path}/geneinfo_beta.txt")
            gene_expr_cp_ch     = Channel.fromPath(params.gctx_cp_path)
            gene_expr_ctl_ch    = Channel.fromPath(params.gctx_ctl_path)


            loadCompoundMetadata(comp_file_ch)
            loadInstanceMetadata(inst_file_ch)
            
            addFingerprints(loadCompoundMetadata.out, params.embedding)
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

            split_data_ch = splitData.out.preprocessed_dataset
    
        }

        if (params.split_dataset_path && params.embedding != "morgan") {
            embedding_data_ch = updateAnnDataFingerprints(split_data_ch, params.embedding)
        } else {
            embedding_data_ch = split_data_ch
        }

        if (params.delete_fp_duplicates) {
            processed_data_ch = delFalseDuplicates(split_data_ch)
        } else {
            processed_data_ch= split_data_ch
        }


        def split_key_ch = Channel.of(0..4).map { fold -> "${params.train_split}_split_${fold}" }   

        training (
            processed_data_ch.first(), 
            split_key_ch, 
            params.smoke_test,
            params.delete_fp_duplicates
        )
    
        testing (
            processed_data_ch.first(),
            training.out.checkpoint,
        )

        publish:

        final_dataset_out         = (params.split_dataset_path && params.embedding == "morgan") ? Channel.empty() : embedding_data_ch

        training_loss_out         = training.out.training_loss
        training_metrics_out      = training.out.training_metrics
        training_checkpoint_out   = training.out.checkpoint

        testing_ctl_out           = testing.out.control
        testing_gt_out            = testing.out.ground_truth
        testing_pred_out          = testing.out.prediction
        testing_split_key_out     = testing.out.split_key_file
        testing_td_splits_out     = testing.out.td_splits
}

output {
    
    final_dataset_out { 
        path "${params.batch}/intermediates" 
        mode 'copy'

    }

    training_loss_out {
        path "${params.batch}/training_${params.train_split}"
        mode 'copy'

    }

    training_metrics_out {
        path "${params.batch}/training_${params.train_split}"
        mode 'copy'

    }

    training_checkpoint_out {
        path "${params.batch}/training_${params.train_split}"
        mode 'copy'

    }

    testing_ctl_out {
        path "${params.batch}/testing_${params.train_split}"
        mode 'copy'

    }
    
    testing_gt_out {
        path "${params.batch}/testing_${params.train_split}"
        mode 'copy'

    }

    testing_pred_out {
        path "${params.batch}/testing_${params.train_split}"
        mode 'copy'

    }

    testing_split_key_out {
        path "${params.batch}/testing_${params.train_split}"
        mode 'copy'

    }

    testing_td_splits_out {
        path "${params.batch}/testing_${params.train_split}"
        mode 'copy'
        
    }
    
}