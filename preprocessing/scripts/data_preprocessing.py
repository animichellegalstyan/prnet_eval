import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os

from preprocessing.scripts.inspect_fingerprints import get_fingerprint, analyze_fingerprint_collision
from pathlib import Path
from pandarallel import pandarallel

def preprocess_metadata(inst_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the raw data by the criteria outlined in the PRnet paper on metadata level. This function:
    - deletes insufficient compound conditions
    - removes invalid compound SMILES
    - pairs perturbed and unperturbed observations, and removes remaining pairless ones

    Parameters
    ----------
    comp_metadata: pd.DataFrame
        A Dataframe containing metadata on the compounds.
    inst_metadata: pd.DataFrame
        A Dataframe containing experimental metadata. 
    
    Returns
    -------
    comp_metadata_clean: pd.DataFrame
        A Dataframe containing metadata on the compounds after cleaning data.
    inst_metadata_clean: pd.DataFrame
        A Dataframe containing experimental metadata after cleaning data. 
    """

    inst_metadata_clean = (inst_metadata
                           .pipe(adapt_cols_to_prnet)
                           .pipe(del_insufficient_comp, verbose=False)
                           .pipe(pair_observations, verbose=False))

    return inst_metadata_clean


def add_fingerprints(comp_metadata: pd.DataFrame, 
                     verbose: False) -> pd.DataFrame:
    """
    Assumes fingerprints have not been generated or need to be generated again 
    (compounds_info_fingerprints.parquet does not exist or is deprecated).

    Deletes rows where SMILES strings are undefined.
    Generates Fingerprints using get_fingerprint from inspect_fingerprints.py and adds them to comp_metadata.
    Deletes rows where Fingerprints are undefined. 

    Saving to parquet with nextflow script

    Parameters
    ----------
    comp_metadata: pd.DataFrame
        A Dataframe containing metadata on the compounds.
    comp_metadata_path: Path
        A Path where comp_metadata with the added fingerprints should be saved.  
    verbose: bool
        Shows number of rows deleted if set to True.

    Returns
    -------
    comp_metadata: pd.DataFrame
        A Dataframe containing metadata on the compounds including the generated Fingerprints.
    """

    # Delete na's in canonical_smiles column
    nrows_before = comp_metadata.shape[0]
    comp_metadata = comp_metadata.dropna(subset=['canonical_smiles'])    
    nrows_after = comp_metadata.shape[0]

    # Generate fingerprints
    pandarallel.initialize(progress_bar=False)
    comp_metadata["fingerprint_smiles"] = comp_metadata["canonical_smiles"].parallel_apply(get_fingerprint)

    # delete na's in fingerprint_smiles column
    comp_metadata = comp_metadata.dropna(subset=['fingerprint_smiles'])
    nrows_after2 = comp_metadata.shape[0]

    # Print information if verbose is True
    if verbose:
        print(f"{nrows_before-nrows_after} rows were deleted due to invalid SMILES.")
        print(f"{nrows_after - nrows_after2} rows remained unparsed by RDKit and deleted.")

    return comp_metadata.reset_index(drop=True)


def del_false_duplicate_fp(comp_metadata: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    # A dictionary to track the verdicts of each fingerprint group
    group_verdicts = {}

    def is_valid_group(group):
        fp = group.name # The fingerprint bitstring for this group
        
        if len(group) <= 1:
            group_verdicts[fp] = "No collision"
            return True
        
        # Get the sorted list of unique reasons for this cluster
        reasons = analyze_fingerprint_collision(smiles_list=group["canonical_smiles"].tolist())
        
        # Save the reasons to our tracker (joined as a string if there are multiple)
        group_verdicts[fp] = ", ".join(reasons)
        
        return reasons == ["True duplicates"]

    # Filter out whole groups based on the logic above
    filtered_metadata = comp_metadata.groupby("fingerprint_smiles", group_keys=False).filter(is_valid_group)
    
    if verbose:
        # Map the group verdicts back to the rows of the ORIGINAL dataframe to count them accurately
        row_reasons = comp_metadata["fingerprint_smiles"].map(group_verdicts)
        
        # Isolate just the rows that were dropped
        dropped_rows = row_reasons[~comp_metadata.index.isin(filtered_metadata.index)]
        
        print("\n=== Drop Reason Breakdown (By Rows Affected) ===")
        print(dropped_rows.value_counts())
        print(f"================================================\nTotal Rows Dropped: {len(dropped_rows)}")

    return filtered_metadata.copy()

def adapt_cols_to_prnet(inst_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Modifies instance metadata: 
        Adds 'Drug', 'cov_drug_name' and 'cov_drug_dose_name' 
        Changes 'cell_iname' to 'cell_type'.
        Adds 'control' column. 

    Parameters
    ----------
    comp_metadata: pd.DataFrame
        A Dataframe containing metadata on the compounds.
    comp_metadata_path: Path
        A Path where comp_metadata with the added fingerprints should be saved.  
    verbose: bool
        Shows number of rows deleted if set to True.

    Returns
    -------
    inst_metadata: pd.DataFrame
        A Dataframe containing the modified instance metadata.
    """

    inst_metadata['Drug'] = inst_metadata['pert_id'].astype(str)
    inst_metadata['cov_drug_name'] = inst_metadata['cell_iname'].astype(str)+ '_' + inst_metadata['Drug'].astype(str)
    inst_metadata['cov_drug_dose_name'] = inst_metadata['cell_iname'].astype(str)+ '_' + inst_metadata['Drug'].astype(str)+ '_' + inst_metadata['pert_dose'].astype(str)

    # change attribute name where cell name is stored to "cell_type" 
    inst_metadata.rename(columns={"cell_iname": "cell_type"}, inplace=True)

    # add control column for easier access
    control = ["ctl_x", "ctl_vehicle", "ctl_untrt", "ctl_vector"] 
    inst_metadata['control'] = inst_metadata['pert_type'].isin(control).astype(int)

    return inst_metadata

def del_insufficient_comp(inst_metadata: pd.DataFrame, verbose: False) -> pd.DataFrame:
    """
    Assumes column 'cov_drug_dose_name' exists (Execute adapt_cols_to_prnet before this).
    Deletes insufficient compound conditions (observations < 5). 
    Handles lack of dose information for perturbed compounds.

    Parameters
    ----------
    inst_metadata: pd.DataFrame
        A Dataframe containing experimental metadata. 
    verbose: bool
        Shows number of rows deleted if set to True.

    Returns
    -------
    inst_metadata: pd.DataFrame
        A Dataframe containing the instance metadata with a sufficient number of observations for use.
    """

    observations_count_df = inst_metadata.groupby('cov_drug_dose_name').size()
    nrows_obs_before = observations_count_df.shape[0]

    observations_count_df = observations_count_df.loc[observations_count_df >= 5]
    nrows_obs_after = observations_count_df.shape[0]

    if verbose:
        print(f"{nrows_obs_before-nrows_obs_after} number of observation were insufficient and removed.")

    if observations_count_df.empty:
        raise ValueError(
            "Empty Dataframe. With the selected filters there are too few observations per perturbation to proceed."
    )

    inst_metadata = inst_metadata[inst_metadata['cov_drug_dose_name'].isin(observations_count_df.index)].copy()

    # Handle Nan's in pert_dose
    inst_metadata_cp = inst_metadata['control'] == 0

    pert_dose_numeric = pd.to_numeric(inst_metadata.loc[inst_metadata_cp, 'pert_dose'], errors='coerce')

    no_dose_availabe = pert_dose_numeric.isna() & inst_metadata_cp

    if no_dose_availabe.any():
        n_dropped = no_dose_availabe.sum()
        if verbose:
            print(f"Dropped {n_dropped} non-control rows due to missing dose.")
    
        inst_metadata = inst_metadata[~no_dose_availabe].copy()
    
    inst_metadata['pert_dose'] = pd.to_numeric(inst_metadata['pert_dose'], errors='coerce').fillna(0.0)

    return inst_metadata.reset_index(drop=True)

def pair_observations(inst_metadata: pd.DataFrame, verbose: False) -> pd.DataFrame:
    """
    Pairs unperturbed and perturbed observations together depending on the cell line.
    Sample ID of the control a perturbed observation maps to is saved in 'paired_control_index'. 
    Deletes perturbed observations without a corresponding control.

    Parameters
    ----------
    inst_metadata: pd.DataFrame
        A Dataframe containing experimental metadata. 
    verbose: bool
        Shows number of rows deleted if set to True.

    Returns
    -------
    inst_metadata: pd.DataFrame
        A Dataframe containing experimental metadata including information on observation pairing. 
    """
    
    # Pair perturbed and unperturbed observations
    for cell_type in inst_metadata.cell_type.unique().tolist():

        matching_controls = inst_metadata[(inst_metadata.control == 1) & (inst_metadata.cell_type == cell_type)]
        
        if len(matching_controls) > 0:
            control_idx = matching_controls['sample_id'].values[0] # PRnet always maps it to first matching control. 
        else:
            control_idx = None  
            print(f"Warning: No control found for cell type: {cell_type}")
    
        inst_metadata.loc[(inst_metadata.cell_type == cell_type), 'paired_control_index'] = control_idx

    # Delete unpaired observations

    nrows_paired_before = inst_metadata.shape[0]

    is_treatment = inst_metadata['control'] == 0
    is_na_index = inst_metadata['paired_control_index'].isna()

    inst_metadata = inst_metadata[~(is_treatment & is_na_index)]

    nrows_paired_after = inst_metadata.shape[0]

    if verbose:
        print(f"{nrows_paired_before-nrows_paired_after} number of observations left unpaired and removed.")

    return inst_metadata.reset_index(drop=True)


if __name__ == "__main__":

    task = sys.argv[1]

    if task == "fingerprint":
            
        input_file  = sys.argv[2]
        output_file = sys.argv[3]

        df_comp = pd.read_parquet(input_file)

        comp_metadata_clean = add_fingerprints(df_comp, verbose=False)
        comp_metadata_clean.to_parquet(output_file, index=False)

    elif task == "preprocess":
            
        input_file  = sys.argv[2]
        output_file = sys.argv[3]
        
        df_inst = pd.read_parquet(input_file)

        inst_metadata_clean = preprocess_metadata(df_inst)
        inst_metadata_clean.to_parquet(output_file, index=False)
    
    elif task == "delete_false_fp_duplicates":
                    
        input_file  = sys.argv[2]
        output_file = sys.argv[3]

        df_split = sc.read_h5ad(input_file)

        filtered_obs = del_false_duplicate_fp(df_split.obs, verbose=True)
        df_split = df_split[filtered_obs.index].copy()
        
    else:
        print(f"Error: Unknown task '{task}'", file=sys.stderr)
        sys.exit(1)
