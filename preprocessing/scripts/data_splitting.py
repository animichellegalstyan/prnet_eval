import anndata as ad
import numpy as np
import scanpy as sc
import sys

from enum import StrEnum
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

class SplittingStrategy(StrEnum):
    COMPOUND_SPLIT_SMILES = "canonical_smiles"
    COMPOUND_SPLIT_PERT_ID = "pert_id"  # just in case they actually split via pert_id
    EMBEDDING_SPLIT = "fingerprint_smiles"

# split_strat is set in python script for now, later will make it selectable from the command line. 
#Split_Setting = str(SplittingStrategy.COMPOUND_SPLIT_SMILES)

def split_data(comp_adata: ad.AnnData, Split_Setting: str, verbose: bool) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    # Split into train / test / val
    train_ratio = 0.6
    validation_ratio = 0.2

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio+validation_ratio, random_state=42)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=train_ratio/(train_ratio+validation_ratio), random_state=41)

    train_and_val_set, test_set = next(gss1.split(comp_adata, groups=comp_adata.obs[Split_Setting]))

    adata_train_val = comp_adata[comp_adata.obs_names[train_and_val_set]].copy()
    adata_test = comp_adata[comp_adata.obs_names[test_set]].copy()

    train_set, val_set = next(gss2.split(adata_train_val, groups=adata_train_val.obs[Split_Setting]))

    train_ids = adata_train_val.obs_names[train_set]
    valid_ids = adata_train_val.obs_names[val_set]


    adata_train = comp_adata[train_ids].copy()
    adata_val = comp_adata[valid_ids].copy()

    if verbose:
        print(f"\n--- Overall Distribution for {Split_Setting} ---")
        print("Training set: ", train_set.size)
        print("Validation set: ", val_set.size)
        print("Testing set: ", test_set.size)
    
    return adata_train, adata_val, adata_test

def split_folds(adata_train: ad.AnnData, Split_Setting: str, verbose: bool) -> ad.AnnData:

    group_5fold_1 = GroupKFold(n_splits=5)
    group_5fold_2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)

    main_groups = adata_train.obs[Split_Setting]

    for fold, (train_and_val_folds, test_folds) in enumerate(
    group_5fold_1.split(adata_train, groups=main_groups)
    ):
        column_name = f"{Split_Setting}_split_{fold}"

        adata_train.obs[column_name] = "train"

        adata_train.obs.iloc[test_folds, adata_train.obs.columns.get_loc(column_name)] = "test"

        adata_train_val_folds = adata_train[train_and_val_folds]

        _, val_folds_subset_idx = next(
            group_5fold_2.split(adata_train_val_folds, groups=adata_train_val_folds.obs["canonical_smiles"])
        )

        # Map the relative validation positions back to the global index space
        global_val_folds = train_and_val_folds[val_folds_subset_idx]

        # Assign 'valid' to those global validation rows
        adata_train.obs.iloc[global_val_folds, adata_train.obs.columns.get_loc(column_name)] = "valid"

    if verbose:
        for i in range(5):
            print(f"\n--- Distribution for {Split_Setting}_split_{i} ---")
            print(adata_train.obs[f"{Split_Setting}_split_{i}"].value_counts())
    
    return adata_train


if __name__ == "__main__":

    # Input ----
    strategy_input = sys.argv[1]
    lincs_adata_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        split_setting = SplittingStrategy(strategy_input)
    except ValueError:
        valid_options = [e.value for e in SplittingStrategy]
        print(f"Error: Invalid strategy '{strategy_input}'. Valid options: {valid_options}", file=sys.stderr)
        sys.exit(1)

    lincs_adata = sc.read_h5ad(lincs_adata_path)
    lincs_adata_cp = lincs_adata[lincs_adata.obs['control'] == 0].copy()

    # Data Spliting ----
    print("Splitting the dataset and preparing for 5-fold cv...")
    lincs_adata = split_data(lincs_adata_cp, split_setting, True)
    lincs_adata = split_folds(lincs_adata_cp, split_setting, True)

    # Output ----
    lincs_adata.write(output_path, compression="gzip")

