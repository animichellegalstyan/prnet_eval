#!/usr/bin/env nextflow

process loadCompoundMetadata {

    input:
    path compound_file

    output:
    path "comp_metadata.parquet"

    script:
    """
    export PYTHONPATH="${projectDir}"
    python -m preprocessing.scripts.data_loading load_comp ${compound_file}
    """
}

process loadInstanceMetadata {

    input:
    path instance_file

    output:
    path "inst_metadata.parquet"

    script:
    """
    export PYTHONPATH="${projectDir}"
    python -m preprocessing.scripts.data_loading load_inst ${instance_file}
    """
}

process addFingerprints {

    input:
    path comp_metadata

    output:
    path "fingerprinted_metadata.parquet"

    script:
    """
    export PYTHONPATH="${projectDir}"
    python -m preprocessing.scripts.data_preprocessing fingerprint ${comp_metadata} fingerprinted_metadata.parquet
    """
}

process delFalseDuplicates {

    input:
    path split_data

    output:
    path "split_data.h5ad"

    script:
    """
    export PYTHONPATH="${projectDir}"
    python -m preprocessing.scripts.data_preprocessing delete_false_fp_duplicates ${split_data} split_data.h5ad
    """
}

process preprocessMetadata {

    input:
    path inst_metadata

    output:
    path "preprocessed_metadata.parquet"

    script:    
    """
    export PYTHONPATH="${projectDir}"
    python -m preprocessing.scripts.data_preprocessing preprocess ${inst_metadata} preprocessed_metadata.parquet
    """
}

process loadExpressionData {

    input:
    path fingerprinted_comp_parquet
    path preprocessed_inst_parquet
    path raw_gene_parquet
    path raw_gctx_cp
    path raw_gctx_ctl

    output:
    path "loaded_dataset.h5ad"
    
    script:
    """
    export PYTHONPATH="${projectDir}"
    
    python -m preprocessing.scripts.data_loading process_expression \
        "${fingerprinted_comp_parquet}" \
        "${preprocessed_inst_parquet}" \
        "${raw_gene_parquet}" \
        "${raw_gctx_cp}" \
        "${raw_gctx_ctl}"
    """

}






