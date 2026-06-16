process training{

    input:
    // adata obj from datasplitting
    path preprocessed_dataset

    output:
    // all the df listed below???
    path "/training/loss_data.h5ad"
    path "/training/metrics_data.h5ad"

    script:
    // call train_lincs
    """
    export PYTHONPATH="${projectDir}"
    python -m train_lincs 
    """

}
        loss_df = pd.DataFrame(loss_dict)
        metrics_df = pd.DataFrame(metrics_dict)
