import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import sys

from enum import StrEnum
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold

class SplittingStrategy(StrEnum):
    COMPOUND_SPLIT_SMILES = "canonical_smiles"
    COMPOUND_SPLIT_PERT_ID = "pert_id"  
    EMBEDDING_SPLIT = "fingerprint_smiles"



def split_data(comp_adata: ad.AnnData, Split_Settings: list[str], verbose: bool) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:

    for split_key in Split_Settings:
        # Split into train / test / val
        train_ratio = 0.6
        validation_ratio = 0.2

        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio+validation_ratio, random_state=42)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=train_ratio/(train_ratio+validation_ratio), random_state=41)

        train_and_val_set, test_set = next(gss1.split(comp_adata, groups=comp_adata.obs[split_key]))

        adata_train_val = comp_adata[comp_adata.obs_names[train_and_val_set]].copy()
        adata_test = comp_adata[comp_adata.obs_names[test_set]].copy()

        train_set, val_set = next(gss2.split(adata_train_val, groups=adata_train_val.obs[split_key]))

        train_ids = adata_train_val.obs_names[train_set]
        valid_ids = adata_train_val.obs_names[val_set]


        adata_train = comp_adata[train_ids].copy()
        adata_val = comp_adata[valid_ids].copy()

        if verbose:
            print(f"\n--- Overall Distribution for {split_key} ---")
            print("Training set: ", train_set.size)
            print("Validation set: ", val_set.size)
            print("Testing set: ", test_set.size)
    
    return adata_train, adata_val, adata_test

def split_folds(adata_train: ad.AnnData, Split_Settings: list[str], verbose: bool) -> ad.AnnData:

    is_control = (adata_train.obs["control"].astype(int) == 1)
    is_compound = ~is_control

    for split_key in Split_Settings:
        group_5fold_1 = GroupKFold(n_splits=5)
        group_5fold_2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)

        adata_compounds = adata_train[is_compound]
        main_groups = (
            adata_compounds.obs[split_key].astype(str) + "_" + 
            adata_compounds.obs["cell_type"].astype(str)
        )

        for fold, (train_and_val_folds, test_folds) in enumerate(
        group_5fold_1.split(adata_compounds, groups=main_groups)
        ):
            column_name = f"{split_key}_split_{fold}"

        for fold, (train_and_val_folds, test_folds) in enumerate(
        group_5fold_1.split(adata_compounds, groups=main_groups)
        ):
            column_name = f"{split_key}_split_{fold}"


            adata_train.obs.loc[is_compound, column_name] = "train"
            adata_train.obs.loc[is_control, column_name] = ""

            test_indices_names = adata_compounds.obs.index[test_folds]
            adata_train.obs.loc[test_indices_names, column_name] = "test"

            adata_train_val_folds = adata_compounds[train_and_val_folds]

            _, val_folds_subset_idx = next(
                group_5fold_2.split(adata_train_val_folds, groups=adata_train_val_folds.obs[split_key])
            )

            val_indices_names = adata_train_val_folds.obs.index[val_folds_subset_idx]
            adata_train.obs.loc[val_indices_names, column_name] = "valid"

        if verbose:
            for i in range(5):
                print(f"\n--- Distribution for {split_key}_split_{i} ---")
                print(adata_train.obs[f"{split_key}_split_{i}"].value_counts())
    
    return adata_train

def split_folds_stratif(adata_train: ad.AnnData, Split_Settings: list[str], verbose: bool) -> ad.AnnData:

    is_control = adata_train.obs["control"].astype(int) == 1
    is_compound = ~is_control

    adata_compounds = adata_train[is_compound]

    for split_key in Split_Settings:
        
        main_groups = adata_compounds.obs[split_key].astype(str)
        strata = adata_compounds.obs["cell_type"].astype(str)

        outer_splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        )

        for fold, (train_and_val_folds, test_folds) in enumerate(
            outer_splitter.split(adata_compounds, y=strata, groups=main_groups)
        ):
            column_name = f"{split_key}_split_{fold}"

            adata_train.obs.loc[is_compound, column_name] = "train"
            adata_train.obs.loc[is_control, column_name] = ""

            test_indices_names = adata_compounds.obs.index[test_folds]
            adata_train.obs.loc[test_indices_names, column_name] = "test"

            adata_train_val = adata_compounds[train_and_val_folds]
            sub_strata = adata_train_val.obs["cell_type"].astype(str)
            sub_groups = adata_train_val.obs[split_key].astype(str)

            inner_splitter = StratifiedGroupKFold(
                n_splits=4, shuffle=True, random_state=42
            )
            _, val_folds_subset_idx = next(
                inner_splitter.split(adata_train_val, y=sub_strata, groups=sub_groups)
            )

            # Assign validation indices
            val_indices_names = adata_train_val.obs.index[val_folds_subset_idx]
            adata_train.obs.loc[val_indices_names, column_name] = "valid"

        if verbose:
            for i in range(5):
                print(f"\n--- Distribution for {split_key}_split_{i} ---")
                print(adata_train.obs[f"{split_key}_split_{i}"].value_counts())

                col = f"{split_key}_split_{i}"
                print(f"\n==========================================")
                print(f"  CELL TYPE DISTRIBUTION: {col}")
                print(f"==========================================")

                # Cross-tabulation of Cell Types vs Split (Train / Valid / Test)
                ct_dist = pd.crosstab(
                    adata_train.obs[is_compound]["cell_type"],
                    adata_train.obs[is_compound][col],
                )

                # Ensure all columns exist even if 0
                for split_name in ["train", "valid", "test"]:
                    if split_name not in ct_dist.columns:
                        ct_dist[split_name] = 0

                ct_dist = ct_dist[["train", "valid", "test"]]
                ct_dist["Total"] = ct_dist.sum(axis=1)

                # Proportional breakdown per cell type (rows sum to 100%)
                ct_pct = (
                    ct_dist[["train", "valid", "test"]].div(
                        ct_dist["Total"], axis=0
                    )
                    * 100
                ).round(1)
                print("\n--- Split Proportions (%) per Cell Type ---")
                print(ct_pct.sort_values(by="train", ascending=False))

    return adata_train

if __name__ == "__main__":

    # Input ----
    strategy_input = sys.argv[1]
    lincs_adata_path = sys.argv[2]
    output_path = sys.argv[3]

    # Check if strategies are valid for splits
    strategy_input = [s.strip() for s in strategy_input.split(',')]
    valid_options = [e.value for e in SplittingStrategy]

    for strategy in strategy_input:
        if strategy not in valid_options:
            print(f"Error: '{strategy}' is not a valid strategy. Choose from: {valid_options}", file=sys.stderr)
            sys.exit(1)

    split_settings = [SplittingStrategy(s) for s in strategy_input]

    lincs_adata = sc.read_h5ad(lincs_adata_path)

    # Data Spliting ----

    print("Splitting the treated dataset and preparing for 5-fold cv...")
    
    #adata_train, adata_val, adata_test = split_data(lincs_adata_cp, split_settings, True)
    adata = split_folds_stratif(lincs_adata, split_settings, True)


    # Output ----
    adata.write(output_path, compression="gzip")



