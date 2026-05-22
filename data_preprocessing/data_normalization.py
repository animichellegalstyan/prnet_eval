import anndata as ad
import scanpy as sc

def preprocessing(clean_data: ad.AnnData) -> ad.AnnData:
    """
    Preprocesses the cleaned data as described in PRnet. The preprocessing is the same for all HTS RNA-seq
    datasets, depending on single-cell and bulk data.

    Parameters
    ----------
    clean_data: ad.AnnData
            AnnData object containing the data after it was cleaned (with data_clean_lincs or data_clean_sciplex) 
    
    Returns
    -------
    preproc_data: ad.Anndata
            AnnData object that containes the preprocessed data. 
    """

    sc.pp.normalize_total(clean_data)
    sc.pp.log1p(clean_data)
    
        


def data_clean_lincs(raw_data: ad.AnnData) -> ad.AnnData:
    """
    Cleans the raw data by the criteria outlined in the PRnet paper. This function:
    - deletes insufficient compound conditions
    - removes invalid compound SMILES
    - pairs perturbed and unperturbed observations, and removes remaining pairless ones

    Parameters
    ----------
    raw_data: ad.AnnData
            AnnData object containing the raw data of the loaded dataset. 
    
    Returns
    -------
    clean_data: ad.AnnData
            AnnData object containing the cleaned data.
    """


