import numpy as np
import pandas as pd
import pytest

from preprocessing.scripts.data_preprocessing import add_fingerprints, adapt_cols_to_prnet, del_insufficient_comp, pair_observations 

# Datasets used for testing ----
sample_fingerprints_df = pd.DataFrame({
    "canonical_smiles": ["CCNC(=O)CCC(N)C(O)=O", "NC(CCCNC(N)=O)C(O)=O", None, "restricted"]
})

sample_adapt_cols_df = pd.DataFrame({
    "pert_id": ["BRD-A61304759", "BRD-K18190982", "BRD-K61567297"],
    "pert_type": ["trt_cp", "ctl_vehicle", "trt_cp"],
    "cell_iname": ["A549", "A549", "TMD8"],
    "pert_dose": [0.001500, 0.015200, 0.156250]
})

sample_array = np.array(["A549_BRD-A61304759_0.001500"]*2 + ["A549_BRD-K18190982_0.015200"]*8)

sample_del_comp_df = pd.DataFrame({
    "cov_drug_dose_name": sample_array.flatten()
})

sample_pair_obs_df = pd.DataFrame({
    "cell_type": ["A549", "A549", "MCF7", "MCF7", "PC3"],
    "control": [0, 1, 0, 1, 0],
    "sample_id": ["EMU001_TMD8_3H_X1_B39:A22", "AICHI002_THP1_4H_X3_B39:C22", "ERG013_PC3_72H_X1_B11:D16", "HSF044_HEK293T_48H_X1_B12:H01", "HSF038_HEK293T_48H_X2_B12:M01"]
})

# Deactivating Parallelization for unit test test_add_fingerprint to strictly test logic. Use preprocessing_notebook to test parallelization ----
pd.Series.parallel_apply = pd.Series.apply
#pd.DataFrame.parallel_apply = pd.DataFrame.apply

# Tests ----
def test_add_fingerprints():

    result = add_fingerprints(sample_fingerprints_df, verbose=False)

    result_smiles = result[["canonical_smiles"]].reset_index(drop=True)

    expectation = pd.DataFrame({
        "canonical_smiles": ["CCNC(=O)CCC(N)C(O)=O", "NC(CCCNC(N)=O)C(O)=O"]
    })

    # canonical_smiles: check if Nan's and unparsable SMILES are removed
    pd.testing.assert_frame_equal(result_smiles, expectation)

    # fingerprint_smiles: check if fingerprints were generated and Nan's removed
    assert 'fingerprint_smiles' in result.columns
    assert result["fingerprint_smiles"].notna().all()


def test_adapt_cols_to_prnet():

    result = adapt_cols_to_prnet(sample_adapt_cols_df)

    expectation = pd.DataFrame({
        "pert_id": ["BRD-A61304759", "BRD-K18190982", "BRD-K61567297"],
        "pert_type": ["trt_cp", "ctl_vehicle", "trt_cp"],
        "cell_type": ["A549", "A549", "TMD8"],
        "pert_dose": [0.001500, 0.015200, 0.156250],
        "Drug": ["BRD-A61304759", "BRD-K18190982", "BRD-K61567297"],
        "cov_drug_name": ["A549_BRD-A61304759", "A549_BRD-K18190982", "TMD8_BRD-K61567297"],
        "cov_drug_dose_name": ["A549_BRD-A61304759_0.0015", "A549_BRD-K18190982_0.0152", "TMD8_BRD-K61567297_0.15625"],
        "control": [0, 1, 0]
    })

    # check if all needed changes have been made
    pd.testing.assert_frame_equal(result, expectation)


def test_del_insufficient_comp():

    result = del_insufficient_comp(sample_del_comp_df, verbose=False)

    expectation = pd.DataFrame({
        "cov_drug_dose_name": np.array(["A549_BRD-K18190982_0.015200"]*8)

    })

    pd.testing.assert_frame_equal(result, expectation)


def test_pair_observations():

    result = pair_observations(sample_pair_obs_df, verbose=False)

    expectation = pd.DataFrame({
        "cell_type": ["A549", "A549", "MCF7", "MCF7"],
        "control": [0, 1, 0, 1],
        "sample_id": ["EMU001_TMD8_3H_X1_B39:A22", "AICHI002_THP1_4H_X3_B39:C22", "ERG013_PC3_72H_X1_B11:D16", "HSF044_HEK293T_48H_X1_B12:H01"],
        "paired_control_index": ["AICHI002_THP1_4H_X3_B39:C22", "AICHI002_THP1_4H_X3_B39:C22", "HSF044_HEK293T_48H_X1_B12:H01", "HSF044_HEK293T_48H_X1_B12:H01"]
    })

    pd.testing.assert_frame_equal(result, expectation)
