import anndata as ad
import numpy as np
import scanpy as sc
import random
from anndata import AnnData

# for simple metrics
from sklearn.metrics import r2_score, mean_squared_error

def control_baseline(adata: AnnData) -> np.ndarray:
    """
    Calculate mean expression per gene across unperturbed control cells.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix

    Returns
    -------
    mean_unperturbed : np.ndarray
        Array of length n_genes with mean expression in control cells
    """

    control_mask = adata.obs["control"] == 1
    if control_mask.sum() == 0:
        raise ValueError("No control cells found.")

    # calculate mean expression per gene across control cells
    control_mean = np.asarray(np.mean(np.stack(control_mask, axis=0), axis=0)).ravel()

    return control_mean

def mean_baseline(adata_train: AnnData) -> np.ndarray[np.float64]:
    """
    Calculate the average gene expression for each cell line
    Note: See how baseline changes if it is dose and time specific as well.

    Parameters 
    ----------
    adata_train : anndata object 
        The annotated data matrix. Contains only the data for training.

    Returns
    -------
    mu_all : np.ndarray[np.float64]
        The mean baseline of adata.
    """
    adata = adata_train[adata_train.obs["control"] == 0]

    aggregated_adata = sc.get.aggregate(adata, by=["cell_type", "pert_id"], func="mean")

    df_baseline = sc.get.obs_df(aggregated_adata, keys=list(aggregated_adata.var_names) + ["cell_type"], layer="mean")

    mean_bl = df_baseline.groupby("cell_type", observed=False).mean()

    return mean_bl

def td_gt_split(adata: ad.AnnData, perturbations: list[str], td_ratio : float = 0.5) -> tuple[dict[str,list[str]],dict[str,list[str]]]:
    """
    For all test perturbations, assign td_ratio% of the cells to a technical duplicate (TD), which corresponds to the mean per gene across these cells.
    The remaining cells are assigned to the ground truth (GT).

    Parameters:
    -----------
    test_perts : list[str]
        A list of all perturbations used for testing (i.E. used for calculating the baseline)
    pert_i_dict : dict[str,list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values
    td_ratio : float, optional
        The percentage of cells used for the technical duplicate (rounded up) (Default: 0.5).

    Returns:
    --------
    td_index_dict : dict[str,list[str]]
        For each perturbed genes as the key, the values are a list of the cells of this perturbation assigned to the technical duplicate.
    gt_index_dict : dict[str, list[str]]
        Same as td_indices, but for the ground truth.
    """    

    skipped_perturbations = []

    td_set = {}
    gt_set = {}

    for pert in perturbations:
        pert_samples = adata.obs_names[adata.obs["pert_id"] == pert].tolist()
        
        # Perturbations with too few cells are skipped:
        if len(pert_samples) < 4:
            skipped_perturbations.append(pert)
            continue
        
        # Split the cells into technical duplicate and ground truth
        random.shuffle(pert_samples)
        td_size = np.ceil(td_ratio*len(pert_samples)).astype(int)
        pert_td_set = np.array(pert_samples[:td_size])
        pert_gt_set = np.array(pert_samples[td_size:])

        td_set[pert] = pert_td_set
        gt_set[pert] = pert_gt_set

    # Print which entries in perturbations were skipped:
    
    if len(skipped_perturbations)>0:
        print("The following perturbations were skipped due to a too low number of cells (<10): ", end = "")
        for p in skipped_perturbations:
            print(p,end="\t")
        print("\n",end="")

    return td_set, gt_set
        
def technical_duplicate(adata: ad.AnnData, td_set : dict[str,list[str]], gt_set : dict[str,list[str]])  -> tuple[dict, dict]:
    """
    Generate the mean expression levels of each gene in the technical duplicate and the ground truth.

    Notes:
    - part set of cells into technical duplicate and ground truth split
    - calculate avg expression profiles for both sets -> technical duplicate avg and ground truth avg
    - IMPORTANT: EXCLUDE CELLS IN TECHNICAL DUPLICATE FROM DEG COMPUTATION TO AVOID DATA LEAKAGE
    Parameters
    ----------
    norm_data : anndata
        The normalized anndata object.    
    td_index_dict : dict[str,list[str]]
        For each perturbed genes as the key, the values are a list of the cells of this perturbation assigned to the technical duplicate.
    gt_index_dict : dict[str, list[str]]
        Same as td_indices, but for the ground truth.

    Returns
    -------
    mu_td_dict : dict[str, np.ndarray[np.float64]]
        A dictionary with the perturbations in test_perts as keys and an array containing the mean of each gene in the technical duplicate cells of the perturbation as values.
    mu_gt_dict : dict[str, np.ndarray[np.float64]]
        Same as mu_td_dict, buth with the mean of each gene in the ground truth cells.
    """
    td_avg_profile = {}
    gt_avg_profile = {}

    for pert in td_set.keys():
        # Calculate the means
        td_matrix = adata[td_set[pert]].X
        #if hasattr(td_matrix, "toarray"): 
        #    td_matrix = td_matrix.toarray()
            
        gt_matrix = adata[gt_set[pert]].X
        #if hasattr(gt_matrix, "toarray"): 
        #    gt_matrix = gt_matrix.toarray()
        avg_td = np.asarray(np.mean(td_matrix, axis=0)).squeeze()
        avg_gt = np.asarray(np.mean(gt_matrix, axis=0)).squeeze()

        # Add the mean counts to the dict
        td_avg_profile[pert] = avg_td
        gt_avg_profile[pert] = avg_gt
    
    return td_avg_profile, gt_avg_profile

def compute_technical_duplicate(adata_split: ad.AnnData, split_name: str, mode: str = "test"):

    mode_data = adata_split[adata_split.obs[split_name] == mode]
    perturbations = mode_data.obs["pert_id"].unique()

    technical_duplicate_set, ground_truth_set = td_gt_split(mode_data, perturbations)

    technical_duplicate_profile, ground_truth_profile = technical_duplicate(mode_data, technical_duplicate_set, ground_truth_set)

    return technical_duplicate_profile, ground_truth_profile

def simple_metrics_mean_bl(mean_baseline: np.ndarray[np.float64], test_compounds: list[str], adata_test: ad.AnnData) -> tuple[np.float64, np.float64]:

    r2_scores_mean_baseline = []
    mse_scores_mean_baseline = []

    for pert in test_compounds:

        # Ground Truth: The actual mean profile of this compound in the test set
        actual_test_matrix = adata_test[adata_test.obs["pert_id"] == pert].X
        yt_m = np.asarray(np.mean(actual_test_matrix, axis=0)).squeeze()
        
        # Prediction: The static baseline vector calculated from the training set
        yp_m = mean_baseline 
        
        r2_scores_mean_baseline.append(r2_score(yt_m, yp_m))
        mse_scores_mean_baseline.append(mean_squared_error(yt_m, yp_m))

    # Calculate mean across all T compounds
    mean_r2 = np.mean(r2_scores_mean_baseline)
    mean_mse = np.mean(mse_scores_mean_baseline)

    return mean_r2, mean_mse

def simple_metrics_td(td_profile: dict[str, np.float64], gt_profile: dict[str, np.float64]) -> tuple[np.float64, np.float64]:
    r2_scores_td = []
    mse_scores_td = []

    for pert in td_profile.keys():
        yt_m = gt_profile[pert] 
        yp_m = td_profile[pert] 

        r2_scores_td.append(r2_score(yt_m, yp_m))
        mse_scores_td.append(mean_squared_error(yt_m, yp_m))
        
    # Calculate mean across all T compounds
    mean_r2 = np.mean(r2_scores_td)
    mean_mse = np.mean(mse_scores_td)

    return mean_r2, mean_mse

def r2_mean(data1, data2):
    sum_r2_1 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2_1 += r2_score_           
    return sum_r2_1/data1.shape[0]

def mse_mean(data1, data2):
    sum_mse_1 = 0
    for i in range(data1.shape[0]):
        mse_score_ = mean_squared_error(data1[i], data2[i])
        sum_mse_1 += mse_score_           
    return sum_mse_1/data1.shape[0]


# Viusalization: Has been generated with Gemini Flash Version 3.5. 
# Updated: 13.07.2026

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import anndata as ad

def plot_performance(fold_results_list: list[dict]):
    all_data_rows = []
    
    for fold_data in fold_results_list:
        fold_idx = fold_data["fold_index"]
        baseline_df = fold_data["baseline_df"]
        adata_test = fold_data["adata_test"]
        gt_profiles = fold_data["gt_profiles"]
        td_profiles = fold_data["td_profiles"]
        model_td_profiles = fold_data.get("model_td_profiles", None)
        
        for pert in gt_profiles.keys():
            if pert not in td_profiles:
                continue
                
            yt_m = gt_profiles[pert]
            yp_td = td_profiles[pert]
            
            pert_adata = adata_test[adata_test.obs["pert_id"] == pert]
            if pert_adata.n_obs == 0:
                continue
            cell_line = pert_adata.obs["cell_type"].iloc[0]
            
            # LEVEL 1: MEAN BASELINE ---
            if cell_line in baseline_df.index:
                yp_baseline = baseline_df.loc[cell_line].values
                r2_bl = r2_score(yt_m, yp_baseline)
                mse_bl = mean_squared_error(yt_m, yp_baseline)
                
                all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "R2 Score", "Value": r2_bl, "Method": "Mean Baseline"})
                all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "MSE", "Value": mse_bl, "Method": "Mean Baseline"})
            
            # LEVEL 2: ML MODEL ---
            if model_td_profiles and pert in model_td_profiles:
                yp_model = model_td_profiles[pert]
                r2_md = r2_score(yt_m, yp_model)
                mse_md = mean_squared_error(yt_m, yp_model)
                
                all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "R2 Score", "Value": r2_md, "Method": "ML Model"})
                all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "MSE", "Value": mse_md, "Method": "ML Model"})
                
            # LEVEL 3: TECHNICAL DUPLICATE ---
            r2_td = r2_score(yt_m, yp_td)
            mse_td = mean_squared_error(yt_m, yp_td)
            
            all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "R2 Score", "Value": r2_td, "Method": "Technical Duplicate"})
            all_data_rows.append({"Fold": f"Fold {fold_idx}", "Compound": pert, "Metric": "MSE", "Value": mse_td, "Method": "Technical Duplicate"})

    df_plot = pd.DataFrame(all_data_rows)
    if df_plot.empty:
        print("No evaluation entries recorded.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    metrics = ["R2 Score", "MSE"]
    colors = {"Mean Baseline": "#e63946", "ML Model": "#457b9d", "Technical Duplicate": "#2a9d8f"}
    has_model = any(df_plot["Method"] == "ML Model")
    plot_order = ["Mean Baseline", "ML Model", "Technical Duplicate"] if has_model else ["Mean Baseline", "Technical Duplicate"]
    
    for i, metric in enumerate(metrics):
        metric_df = df_plot[df_plot["Metric"] == metric]
        
        # Clean Boxplot only: Assigned hue="Method" and legend=False to prevent future warnings
        sns.boxplot(
            data=metric_df, 
            x="Method", 
            y="Value", 
            hue="Method",
            ax=axes[i], 
            palette=colors,
            width=0.4,
            order=plot_order,
            legend=False
        )
        
        # Calculate cross-validation average text overlays
        for method in metric_df["Method"].unique():
            if method in plot_order:
                pos_idx = plot_order.index(method)
                m_val = metric_df[metric_df["Method"] == method]["Value"].mean()
                axes[i].text(
                    pos_idx, m_val, f"CV Mean:\n{m_val:.4f}", 
                    weight='bold', ha='center', va='bottom', color='black', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )
                
        axes[i].set_title(f"5-Fold CV Distribution of {metric}", fontsize=14, pad=15)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].set_xlabel("Evaluation Baseline / Model", fontsize=12)

    plt.tight_layout()
    plt.show()