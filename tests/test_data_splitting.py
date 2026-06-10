import anndata as ad
import numpy as np
import pandas as pd
import pytest

from preprocessing.scripts.data_splitting import Split_Setting, split_data, split_folds

@pytest.fixture
def mock_adata():
    """
    Creates a minimal 10x2 AnnData object for unit testing data splits.
    
    - Rows (obs): Sample IDs containing chemical and biological metadata.
    - Columns (var): Gene IDs containing symbol and title metadata.
    """
    X = np.zeros((10, 2), dtype=np.float32) # Matrix content can remain the same
    
    # Observables (obs) - Updated to have exactly 5 distinct groups
    obs = pd.DataFrame({
        "pert_id": [
            "BRD-A", "BRD-A", 
            "BRD-B", "BRD-B", 
            "BRD-C", "BRD-C", 
            "BRD-D", "BRD-D", 
            "BRD-E", "BRD-E"
        ],
        "canonical_smiles": [
            "CCO", "CCO",    # Group 1
            "CCN", "CCN",    # Group 2
            "CCF", "CCF",    # Group 3
            "CCCl", "CCCl",  # Group 4
            "CCBr", "CCBr"   # Group 5 (Added to satisfy 5 splits)
        ],
        "fingerprint_smiles": [
            "101010", "101010",
            "010101", "010101",
            "110011", "110011",
            "001100", "001100",
            "111000", "111000"   # New matching fingerprint
        ]
    }, index=[f"sample_{i:02d}" for i in range(1, 11)])
    
    var = pd.DataFrame({
        "gene_symbol": ["BRCA1", "TP53"],
        "gene_title": ["BRCA1 DNA repair associated", "Tumor protein p53"]
    }, index=["gene_7157", "gene_2414"])
    
    return ad.AnnData(X=X, obs=obs, var=var)

def test_split_data(mock_adata):

    result_train, result_val, result_test = split_data(mock_adata, verbose=False)

    # check if after split sum of rows is the same as before
    assert mock_adata.shape[0] == (result_train.shape[0]+result_val.shape[0]+result_test.shape[0])

    # check if split ratio is adequate in light of grouping (6:2:2)
    train_ratio_result = result_train.shape[0]/mock_adata.shape[0] 
    val_ratio_result = result_val.shape[0]/mock_adata.shape[0]
    test_ratio_result = result_test.shape[0]/mock_adata.shape[0]

    assert np.isclose(train_ratio_result, 0.6, atol=0.21)
    assert np.isclose(val_ratio_result, 0.2, atol=0.21)
    assert np.isclose(test_ratio_result, 0.2, atol=0.15)

    # check for data leakage
    assert set(result_train.obs[Split_Setting]).isdisjoint(set(result_val.obs[Split_Setting]))
    assert set(result_train.obs[Split_Setting]).isdisjoint(set(result_test.obs[Split_Setting]))
    assert set(result_val.obs[Split_Setting]).isdisjoint(set(result_test.obs[Split_Setting]))


def test_split_folds_columns(mock_adata):

    result = split_folds(mock_adata, verbose=False)

    # check if data split columns exist
    for fold in range(5):
        assert f"drug_split_{fold}" in result.obs

def test_split_folds_ratio(mock_adata):

    result = split_folds(mock_adata, verbose=False)

    # check if split ratio within each fold is adequate in light of grouping (6:2:2)
    for fold in range(5):
        result_train = result[result.obs[f"drug_split_{fold}"]=="train"]
        result_val = result[result.obs[f"drug_split_{fold}"]=="valid"]
        result_test = result[result.obs[f"drug_split_{fold}"]=="test"]

        train_ratio_result = result_train.shape[0]/mock_adata.shape[0] 
        val_ratio_result = result_val.shape[0]/mock_adata.shape[0]
        test_ratio_result = result_test.shape[0]/mock_adata.shape[0]

        assert np.isclose(train_ratio_result, 0.6, atol=0.21)
        assert np.isclose(val_ratio_result, 0.2, atol=0.21)
        assert np.isclose(test_ratio_result, 0.2, atol=0.15)

def test_split_folds_leakage(mock_adata):

    result = split_folds(mock_adata, verbose=False)

    # check for data leakage within each fold
    for fold in range(5):
        result_train = result[result.obs[f"drug_split_{fold}"]=="train"]
        result_val = result[result.obs[f"drug_split_{fold}"]=="valid"]
        result_test = result[result.obs[f"drug_split_{fold}"]=="test"]

        assert set(result_train.obs[Split_Setting]).isdisjoint(set(result_val.obs[Split_Setting]))
        assert set(result_train.obs[Split_Setting]).isdisjoint(set(result_test.obs[Split_Setting]))
        assert set(result_val.obs[Split_Setting]).isdisjoint(set(result_test.obs[Split_Setting]))