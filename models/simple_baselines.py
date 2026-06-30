import anndata as ad
import numpy as np
import scanpy as sc
import random
from anndata import AnnData

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

def mean_baseline(adata: AnnData) -> np.ndarray[np.float64]:
    """
    Calculate the average gene expression counts for all genes for each perturbation. 
    Then compute the mean expressions of each gene across all perturbations

    Parameters 
    ----------
    norm_data : anndata object 
        The annotated data matrix
    pert_i_dict : dict[str, list[str]]
        A dictionary containing the names of perturbations as keys and the indices of cells in norm_data associated with this perturbation as values
    train_perts : list[str]
        A list of all perturbations used for testing (i.E. used for calculating the baseline)

    Returns
    -------
    mu_all : np.ndarray[np.float64]
        The mean baseline of norm_data for a specified layer
    """
    aggregated_adata = sc.get.aggregate(adata, by="pert_id", func="mean")

    mu_all = np.mean(aggregated_adata.X, axis=0)

    return mu_all

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

def compute_technical_duplicate(adata_split: ad.AnnData, split_name: str, mode: str = "train"):

    mode_data = adata_split[adata_split.obs[split_name] == mode]
    perturbations = mode_data.obs["pert_id"].unique()

    technical_duplicate_set, ground_truth_set = td_gt_split(mode_data, perturbations)

    technical_duplicate_profile, ground_truth_profile = technical_duplicate(mode_data, technical_duplicate_set, ground_truth_set)

    return technical_duplicate_profile, ground_truth_profile