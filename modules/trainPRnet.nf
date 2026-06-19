process training{

    conda 'prnet_run_env'

    input:
    path preprocessed_dataset
    val split_key
    val smoke_test

    output:
    path "loss_data.h5ad", emit: training_loss
    path "metrics_data.h5ad", emit: training_metrics

    script:
    def test_flag = smoke_test ? "--smoke_test" : ""
    """
    export PYTHONPATH="${projectDir}"
    python -m train_lincs --input_data ${preprocessed_dataset} --split_key ${split_key} ${test_flag}
    """

}
