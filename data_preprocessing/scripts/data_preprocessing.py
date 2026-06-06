import anndata as ad
import pandas as pd
import scanpy as sc

from data_preprocessing.scripts.inspect_fingerprints import get_fingerprint
from pathlib import Path


def preprocessing(comp_metadata_clean: pd.DataFrame, inst_metadata_clean: pd.DataFrame) -> ad.AnnData:
    """
    Preprocesses the cleaned data as described in PRnet. The preprocessing is the same for all HTS RNA-seq
    datasets, depending on single-cell and bulk data.

    Parameters
    ----------
    cleaned_data: ad.AnnData
        AnnData object containing the data after it was cleaned (with data_clean_lincs or data_clean_sciplex) 
    
    Returns
    -------
    preprocessed_data: ad.Anndata
        AnnData object that containes the preprocessed data. 
    """
    preprocessed_data = sc.pp.normalize_total(cleaned_data)
    preprocessed_data = sc.pp.log1p(cleaned_data)
    
    return preprocessed_data
    
        


def data_cleaning(comp_metadata: pd.DataFrame, 
                  inst_metadata: pd.DataFrame,
                  comp_metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    comp_metadata_clean = add_fingerprints(comp_metadata, comp_metadata_path)
    inst_metadata_clean = (inst_metadata
                           .pipe(adapt_cols_to_prnet)
                           .pipe(del_insufficient_comp)
                           .pipe(pair_observations))



    inst_meta_prnet_cols = adapt_cols_to_prnet(inst_metadata)
    inst_meta_sufficient = del_insufficient_comp(inst_meta_prnet_cols)
    inst_metadata_clean = pair_observations(inst_meta_sufficient)

    return comp_metadata_clean, inst_metadata_clean

def add_fingerprints(comp_metadata: pd.DataFrame, comp_metadata_path: Path, verbose: bool) -> pd.DataFrame:
    """
    Assumes fingerprints have not been generated or need to be generated again 
    (compounds_info_fingerprints.parquet does not exist or is deprecated).

    Deletes rows where SMILES strings are undefined.
    Generates Fingerprints using get_fingerprint from inspect_fingerprints.py and adds them to comp_metadata.
    Deletes rows where Fingerprints are undefined. 

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
    comp_metadata["Fingerprint_smiles"] = comp_metadata["canonical_smiles"].parallel_apply(get_fingerprint)

    # Save metdata to Parquet
    comp_metadata.to_parquet(comp_metadata_path, index=False)

    # delete na's in fingerprint_smiles column
    comp_metadata = comp_metadata.dropna(subset=['fingerprint_smiles'])
    nrows_after2 = comp_metadata.shape[0]

    # Print information if verbose is True
    if verbose:
        print(f"{nrows_before-nrows_after} rows were deleted due to invalid SMILES.")
        print(f"{nrows_after - nrows_after2} rows remained unparsed by RDKit and deleted.")

    return comp_metadata

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

def del_insufficient_comp(inst_metadata: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """
    Assumes column 'cov_drug_dose_name' exists (Execute adapt_cols_to_prnet before this).
    Deletes insufficient compound conditions (observations < 5). 

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

    return inst_metadata

def pair_observations(inst_metadata: pd.DataFrame, verbose: bool) -> pd.DataFrame:
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

    nrows_paired_before = inst_info_merged.shape[0]

    is_treatment = inst_info_merged['pert_type'] == "trt_cp"
    is_na_index = inst_info_merged['paired_control_index'].isna()

    inst_info_merged = inst_info_merged[~(is_treatment & is_na_index)]

    nrows_paired_after = inst_info_merged.shape[0]

    if verbose:
        print(f"{nrows_paired_before-nrows_paired_after} number of observations left unpaired and removed.")

    return inst_metadata