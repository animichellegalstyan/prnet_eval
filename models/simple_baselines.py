import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import random

from anndata import AnnData
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from typing import Dict, List


def control_baseline(adata: AnnData, split_axis: str) -> pd.DataFrame:
    """
    Calculates mean gene expression profile of control samples for each fold.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix. 

    split_axis: str
        A string describing along which axis data was split on. 
        Options are 'canonical_smiles' and 'fingerprint_smiles'

    Returns
    -------
    ctl_expression_profile_df : pd.DataFrame
        A dataframe containing the mean gene expression profile of control cells for each fold.
        rows: folds
        columns: genes
    """
    ctl_baseline_matrix = {}

    folds = range(0,5)
    for fold in folds:
        control_data = adata[adata.obs['control'] == 1]

        ctl_expression_profile = control_data.X.mean(axis=0)
        ctl_baseline_matrix[fold] = np.ravel(ctl_expression_profile)
    
    ctl_expression_profile_df = pd.DataFrame.from_dict(
        ctl_baseline_matrix, orient="index", columns=adata.var_names
    )
    ctl_expression_profile_df.index.name = f"{split_axis} fold"
    ctl_expression_profile_df.columns.name = None

    return ctl_expression_profile_df


def mean_baseline_fold(fold_train: AnnData, specification: str) -> pd.DataFrame:
    """
    Helper function for mean_baseline.
    Calculates mean gene expression profile of perturbed samples for a given fold.
    The expression profile will be specific for the specification given.

    Note: See how baseline changes if it is dose and time specific as well.

    Parameters 
    ----------
    fold_train : anndata object 
        The annotated data matrix. Contains only the data for training.
    specification : str
        A string decribing which attribute the mean expression profile should be specific towards. 
        Options are any attribute of adata.obs

    Returns
    -------
    mean_expression_profile_df : pd.DataFrame
        A dataframe containing the mean gene expression profile of perturbed cells for each fold. 
        rows: folds
        columns: genes
    """

    aggregated_adata = sc.get.aggregate(fold_train, by=[specification, "pert_id"], func="mean")

    df_baseline = sc.get.obs_df(aggregated_adata, keys=list(aggregated_adata.var_names) + [specification], layer="mean")

    mean_expression_profile_df = df_baseline.groupby(specification, observed=False).mean()

    return mean_expression_profile_df

def mean_baseline(adata_train: AnnData, cell_line_dict: Dict[int, List[str]], specification: str, split_axis: str):
    """
    Computes mean baseline for every fold.

    Parameters
    ----------
    adata_train : anndata.AnnData
        The annotated data matrix.
    cell_line_dict:
        A dictionary containing the number of the fold as key and a list of unique cell_line names as values. 
        The dictionary helps to identify the cell lines which were a part of testing the model.
    specification : str
        A string decribing which attribute the mean expression profile should be specific towards. 
        Options are any attribute of adata.obs
    split_axis: str
        A string describing along which axis data was split on. 
        Options are 'canonical_smiles' and 'fingerprint_smiles'
    
    Returns
    -------
    mean_expression_profile_df : pd.DataFrame
        A dataframe containing the mean gene expression profile of perturbed cells for each fold. 
        rows: folds
        columns: genes, fold name
    """

    folds = range(0,5)
    mean_baseline_matrix = []

    for fold in folds:

        fold_train = adata_train[adata_train.obs[f'{split_axis}_split_{fold}'] == 'train']
        fold_train = fold_train[fold_train.obs['cell_type'].isin(cell_line_dict[fold])]

        fold_profile = mean_baseline_fold(fold_train, specification)

        fold_profile['fold name'] = f'{split_axis}_split_{fold}'

        mean_baseline_matrix.append(fold_profile)

    mean_expression_profile_df = pd.concat(mean_baseline_matrix, axis=0)

    mean_expression_profile_df.index.name = f"{specification}"
    mean_expression_profile_df.columns.name = None

    return mean_expression_profile_df


def technical_duplicate_fold(fold: AnnData, td_split_df: pd.DataFrame, fold_name: str) -> pd.DataFrame:


    meta_list = []
    expression_list = []

    pert_groups = td_split_df.groupby('pert_id', observed=False)
    
    for pert_id, group_df in pert_groups:
        
        gt_indices = group_df.index[group_df["ground_truth"] == 1]
        td_indices = group_df.index[group_df["technical_duplicate"] == 1]

        
        # td_split_df has filtered out nans
        
        gt_indices = gt_indices.intersection(fold.obs.index)
        td_indices = td_indices.intersection(fold.obs.index)

        if len(gt_indices) == 0 or len(td_indices) == 0:
            continue
        
        gt_avg_expression_profile = np.ravel(fold[gt_indices, :].X.mean(axis=0))
        td_avg_expression_profile = np.ravel(fold[td_indices, :].X.mean(axis=0))

        meta_list.extend([
            group_df.loc[gt_indices].iloc[0].to_dict(),
            group_df.loc[td_indices].iloc[0].to_dict()
        ])
        expression_list.extend([gt_avg_expression_profile, td_avg_expression_profile])

    meta_df = pd.DataFrame(meta_list)
    expr_df = pd.DataFrame(expression_list, columns=fold.var_names.values)

    td_baseline_df = pd.concat([meta_df, expr_df], axis=1)

    return td_baseline_df


def technical_duplicate_baseline(adata_dict: Dict[int, AnnData], td_split_df: pd.DataFrame, split_type: str, split_axis: str) -> pd.DataFrame:

    folds = range(0,5)
    technical_duplicate_matrix = []

    for fold in folds:
        adata = adata_dict[fold]

        split_name = f'{split_axis}_split_{fold}'
        td_split_df_fold = td_split_df[td_split_df['fold'] == fold]
        
        fold_profile = technical_duplicate_fold(fold=adata, td_split_df=td_split_df_fold, fold_name=split_name)

        fold_profile['fold name'] = f'{split_axis}_split_{fold}'

        technical_duplicate_matrix.append(fold_profile)

    technical_duplicate_df = pd.concat(technical_duplicate_matrix, axis=0)

    return technical_duplicate_df


def td_split(fold_test: ad.AnnData, fold_name: str, td_size: float=0.20) -> pd.DataFrame:
    is_control = fold_test.obs["control"] == 1
    is_compound = ~is_control
    
    fold = fold_test[is_compound]

    strata = (
        fold.obs["cell_type"].astype(str) + "___" + fold.obs["pert_id"].astype(str)
    )

    counts = strata.value_counts()
    valid_mask = strata.isin(counts[counts >= 2].index)
    
    singletons = fold[~valid_mask]

    gt_samples = set()
    td_samples = set()

    if valid_mask.any():
        fold_valid = fold[valid_mask]
        strata_valid = strata[valid_mask]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=td_size, random_state=42)
        gt_idx_local, td_idx_local = next(splitter.split(fold_valid, strata_valid))

        gt_samples.update(fold_valid.obs.index[gt_idx_local])
        td_samples.update(fold_valid.obs.index[td_idx_local])

    gt_samples.update(singletons.obs.index)
    td_split_df = pd.DataFrame(
        index=fold.obs.index,
        data={
            "sample_id": fold.obs.index,
            "cell_type": fold.obs["cell_type"],
            "pert_id": fold.obs["pert_id"],
            "ground_truth": fold.obs.index.isin(gt_samples).astype(int),
            "technical_duplicate": fold.obs.index.isin(td_samples).astype(int),
            "fold": fold_name
        },
    )

    return td_split_df