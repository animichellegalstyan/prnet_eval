import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from data._utils import contribution_df


def load_test_results(results_path: Path, split_key: str, preprocessed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Loads the results generated from testing the model.

    Parameters
    ----------
    results_path: Path
        The path to the folder where all test results are stored.
    split_key: str
        Describes the splitting strategy used for the testing (and training) of the model, including the number of the fold
        to load. Example: 'canonical_smiles_split_0'
    preprocessed_df: pd.Dataframe
        The observables dataframe from the preprocessed anndata object used for training and testing this split.

    returns
    -------
    control_df: pd.DataFrame
        A Dataframe containing the control data
    all_true_df: pd.DataFrame
        A Dataframe containing the ground truth of the expression of perturbed genes.
    all_pre_df: pd.DataFrame 
        A Dataframe containing the predicted expression of perturbed genes.
    """
    """
    pre_array = np.genfromtxt(results_path / f'{split_key}_y_pre_array.csv', delimiter=',')
    true_array = np.genfromtxt(results_path / f'{split_key}_y_true_array.csv', delimiter=',')
    control_array = np.genfromtxt(results_path / f'{split_key}_x_true_array.csv', delimiter=',')
    """

    pre_df = pd.read_csv(results_path / f'{split_key}_y_pre_array.csv', header=None, engine='pyarrow')
    true_df = pd.read_csv(results_path / f'{split_key}_y_true_array.csv', header=None, engine='pyarrow')
    control_df = pd.read_csv(results_path / f'{split_key}_x_true_array.csv', header=None, engine='pyarrow')

    f = open(results_path / f'{split_key}_cov_drug_array.csv',"r")
    lines = f.readlines()
    cov_drug = [x.strip() for x in lines]

    if len(cov_drug) != len(pre_df):
        # Taking exact length if appended sequentially, or deduplicating preserving order:
        cov_drug = cov_drug[:len(pre_df)]

    pre_df.index = cov_drug
    true_df.index = cov_drug
    control_df.index = cov_drug

    is_control = pd.Series(cov_drug).str.lower().isin(["control", "vehicle"]).values
    is_not_control = ~is_control

    pre_df = pre_df[is_not_control]
    true_df = true_df[is_not_control]
    control_df = control_df[is_control]

    pre_df = contribution_df(pre_df)
    true_df = contribution_df(true_df)
    control_df = contribution_df(control_df)


    return control_df, pre_df, true_df

def control_baseline_metrics(adata: ad.AnnData, ctl_expression_profile: pd.DataFrame) -> pd.DataFrame:

    all_fold_metrics = []

    for fold_num in range(5):
        all_fold_metrics = []    
        data = adata.obs[f'canonical_smiles_split_{fold_num}'] == 'test'
        fold_obs = adata.obs[data]
        fold_X = adata[data, :].X
        

        predicted_control_profile = ctl_expression_profile.loc[fold_num].to_numpy()

        for index, sample_id in enumerate(fold_obs.index):
            sample_expression_profile = fold_X[index].toarray().ravel()

            pearson_ctl = pearsonr(x=sample_expression_profile, y=predicted_control_profile).correlation
            pearson_delta_ctl = 0.0 if (predicted_control_profile == predicted_control_profile).all() else pearsonr(x=sample_expression_profile-predicted_control_profile, y=predicted_control_profile-predicted_control_profile).correlation
            mse_ctl = mean_squared_error(y_true=sample_expression_profile, y_pred=predicted_control_profile)

            all_fold_metrics.append(
                {
                    "fold": fold_num,
                    "sample_id": sample_id,
                    "Pearson": pearson_ctl,
                    "Pearson Delta": pearson_delta_ctl,
                    "MSE": mse_ctl,
                }
            )

    control_metrics = pd.DataFrame(all_fold_metrics)

    return control_metrics

def mean_baseline_metrics(adata: ad.AnnData, cell_line_dict: dict[int, ad.AnnData], mean_expression_profile: pd.DataFrame, ctl_expression_profile: pd.DataFrame, split_axis: str) -> pd.DataFrame:

    all_fold_metrics = []

    for fold_num in range(5):
        split_name = f"{split_axis}_split_{fold_num}"
        
        mask = adata.obs[split_name] == 'train'

        cell_lines_dict_fold = cell_line_dict[fold_num]
        test_cell_lines = adata.obs['cell_type'].isin(cell_lines_dict_fold)
        
        fold_obs = adata.obs[mask & test_cell_lines]

        mean_baseline_fold = mean_expression_profile[mean_expression_profile["fold name"] == split_name].drop(columns=["fold name"])
        control_profile = ctl_expression_profile.loc[fold_num].to_numpy()

        fold_X = adata[mask & test_cell_lines, :].X

        sample_ids = fold_obs.index
        cell_types = fold_obs["cell_type"].values

        for index, sample_id in enumerate(sample_ids):
            sample_cell_type = cell_types[index]
            sample_expression_profile = fold_X[index].toarray().ravel()

            predicted_mean_profile = mean_baseline_fold.loc[sample_cell_type].to_numpy()

            mse_mean = mean_squared_error(sample_expression_profile, predicted_mean_profile)
            r2_mean = pearsonr(x=sample_expression_profile, y=predicted_mean_profile).correlation
            pearson_delta_mean = 0.0 if (predicted_mean_profile == control_profile).all() else pearsonr(x=sample_expression_profile-control_profile, y=predicted_mean_profile-control_profile).correlation
            all_fold_metrics.append(
                {
                    "fold": fold_num,
                    "sample_id": sample_id,
                    "cell_type": sample_cell_type,
                    "Pearson": r2_mean,
                    "Pearson Delta": pearson_delta_mean,
                    "MSE": mse_mean,
                }
            )

    mean_bl_metrics_smiles = pd.DataFrame(all_fold_metrics)

    return mean_bl_metrics_smiles

def technical_duplicate_metrics(adata_dict: dict[int, ad.AnnData], technical_duplicate: pd.DataFrame, ctl_expression_profile: pd.DataFrame, split_axis: str):
    all_fold_metrics = []

    for fold_num in range(5):
        adata = adata_dict[fold_num]
        split_name = f"{split_axis}_split_{fold_num}"
        
        mask = adata.obs[split_name] == 'test'
        fold_obs = adata.obs[mask]
        if len(fold_obs) == 0:
            continue

        control_profile = ctl_expression_profile.loc[fold_num].to_numpy()

        # 3. Build technical duplicate & ground truth lookups for this fold
        td_lookup = (
            technical_duplicate[(technical_duplicate["fold name"] == split_name) & (technical_duplicate["technical_duplicate"] == 1)]
            .set_index("pert_id")
            .drop(columns=["technical_duplicate", "ground_truth", "fold name", "fold"])
        )

        gt_lookup = (
            technical_duplicate[(technical_duplicate["fold name"] == split_name) & (technical_duplicate["ground_truth"] == 1)]
            .set_index("pert_id")
            .drop(columns=["technical_duplicate", "ground_truth", "fold name", "fold"])
        )

        valid_perts = set(td_lookup.index).intersection(gt_lookup.index)

        # 4. Extract arrays once per fold (fast NumPy indexing)
        sample_ids = fold_obs.index
        pert_ids = fold_obs["pert_id"].values

        for index, sample_id in enumerate(sample_ids):
            sample_pert = pert_ids[index]

            if sample_pert in valid_perts:
                td_profile = td_lookup.loc[sample_pert].to_numpy().flatten()
                gt_profile = gt_lookup.loc[sample_pert].to_numpy().flatten()

                mse_td = mean_squared_error(gt_profile, td_profile)
                r2_td = pearsonr(x=gt_profile, y=td_profile).correlation
                pearson_delta_td = 0.0 if (td_profile == control_profile).all() else pearsonr(x=gt_profile - control_profile, y=td_profile - control_profile).correlation
            else:
                r2_td = np.nan
                pearson_delta_td = np.nan
                mse_td = np.nan

            all_fold_metrics.append(
                {
                    "fold": fold_num,
                    "sample_id": sample_id,
                    "pert_id": sample_pert,
                    "Pearson": r2_td,
                    "Pearson Delta": pearson_delta_td,
                    "MSE": mse_td,
                }
            )

    td_metrics_smiles = pd.DataFrame(all_fold_metrics)

    return td_metrics_smiles



def prnet_evaluation_metrics(ctl_profile: pd.DataFrame, all_true_df: pd.DataFrame, all_pre_df: pd.DataFrame, split_key: str):
    """
    Computes metrics pearsonr, pearsonr delta and mse for the prnet test results.

    Parameters
    ----------
    control_df: pd.DataFrame
        A Dataframe contraining the control data
    all_true_df: pd.DataFrame
        A Dataframe contraining the ground truth of the expression of perturbed genes.
    all_pre_df: pd.DataFrame 
        A Dataframe contraining the predicted expression of perturbed genes.
    split_key: str
        Describes the splitting strategy used for the testing (and training) of the model, including the number of the fold
        to load. Example: 'canonical_smiles_split_0'
    results_path: Path
        The path to the folder where all test results are stored.

    returns
    -------
    The metrics will be saved in a csv-file in the same directory as given by results_path and split_key.
    """
    
    cols_to_drop = ["cell_type", "condition", "cov_drug_name"]

    true_num = all_true_df.drop(columns=cols_to_drop)
    pre_num = all_pre_df.drop(columns=cols_to_drop)


    # 2. Rename the index to 0..977
    #ctl_profile.index = list(range(len(ctl_profile)))

    metrics_dict = []
    cov_drug_ids = all_true_df.index
    for index, cov_drug_id in enumerate(cov_drug_ids):

        sample_expression_profile = true_num.iloc[index].to_numpy()
        predicted_profile = pre_num.iloc[index].to_numpy()

        mse = mean_squared_error(sample_expression_profile, predicted_profile)
        pearson = pearsonr(x=sample_expression_profile, y=predicted_profile).correlation
        pearson_delta = 0.0 if (predicted_profile == ctl_profile).all() else pearsonr(x=sample_expression_profile-ctl_profile, y=predicted_profile-ctl_profile).correlation

        metrics_dict.append(
            {
                "fold": split_key,
                "cov_drug": cov_drug_id,
                "Pearson": pearson,
                "Pearson Delta": pearson_delta,
                "MSE": mse,
            }
        )
    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df