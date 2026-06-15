# Imports -----

import anndata as ad
import h5py
import os
import pandas as pd
import sys
import time
import warnings

from cmapPy.pandasGEXpress.parse import parse # for gctx
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
from scipy import sparse
from typing import Any, Dict, List, Optional, Union

# Load metadata ----

def load_metadata_txt(file_path: Path, delimiter="\t") -> pd.DataFrame:
    """
    Loads metadata txt files of the LINCS dataset and returns a dataframe.
    """
    with open(file_path, "r") as f:
        raw = f.readlines()
    splitted = [row.strip().split(delimiter) for row in raw]
    df = pd.DataFrame(data=splitted[1:], columns=splitted[0])

    # Sanitize
    df.replace("-666", None, inplace=True)
    df.replace(-666, None, inplace=True)
    df.replace('""', None, inplace=True)

    numeric_cols = ["pert_dose", "pert_time"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_lincs_meta(metadata_folder_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    metadata_folder_path = Path(metadata_folder_path)

    comp_info_merged = load_metadata_txt(metadata_folder_path / "compoundinfo_beta.txt")
    gene_info_merged = load_metadata_txt(metadata_folder_path / "geneinfo_beta.txt")
    inst_info_merged = load_metadata_txt(metadata_folder_path / "instinfo_beta.txt")

    return comp_info_merged, gene_info_merged, inst_info_merged

# Load GE from LINCS and make into anndata object ----
def load_ge(inst_metadata: pd.DataFrame,
            comp_metadata: pd.DataFrame,
            gene_metadata: pd.DataFrame,
            gctx_cp_path: Path,
            gctx_ctl_path: Path) -> ad.AnnData:
    
    # Subset metadata into control and compound
    inst_info_merged_control = inst_metadata[inst_metadata["control"] == 1]
    inst_info_merged_comp = inst_metadata[inst_metadata["pert_type"] == "trt_cp"]

    # Inistantiate the dataloaders
    dataloader_cp = LINCSDataLoader(
        gctx_path=gctx_cp_path,
        inst_info=inst_info_merged_comp,
        gene_info=gene_metadata,
        gene_marker="landmark",
        comp_identifier="pert_id",  
        cell_identifier="cell_type",
        instance_identifier="sample_id",
    )

    control_types = ["ctl_x", "ctl_vehicle", "ctl_untrt", "ctl_vector"] 
    dataloader_ctl = LINCSDataLoader(
        gctx_path=gctx_ctl_path,
        inst_info=inst_info_merged_control,
        gene_info=gene_metadata,
        gene_marker="landmark",
        comp_identifier="pert_id",
        cell_identifier="cell_type",
        instance_identifier="sample_id",
        pert_types=control_types,
    )

    inst_filters_ctl = {}
    inst_filters_comp = {}

    gene_filters_ctl = {}
    gene_filters_comp = {}

    lincs_adata_ctl = dataloader_ctl.create_anndata(False, comp_metadata, inst_filters_ctl)
    lincs_adata_cp = dataloader_cp.create_anndata(False, comp_metadata, inst_filters_comp)

    lincs_adata = ad.concat([lincs_adata_ctl, lincs_adata_cp], axis='obs', join='outer', merge='same')

    return lincs_adata


# Class to help load GE from LINCS datasets ----

class Timer:
    def __init__(self, name: str):
        self.name: str = name
        self.total_seconds: float = 0.0
        self._start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.total_seconds += elapsed
            # Resetting to None prevents double-counting if exit is called twice
            self._start_time = None

    def __str__(self) -> str:
        """Returns a human-readable string of the total time."""
        minutes, seconds = divmod(self.total_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0:
            parts.append(f"{int(minutes)}m")
        parts.append(f"{seconds:.2f}s")

        return f"{self.name}: {' '.join(parts)}"

    def print_restart(self) -> None:
        print(self)
        self.total_seconds = 0.0

class LINCSDataLoader(BaseModel):
    # Pydantic doesn't natively parse DataFrames => explicitly allow them
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 1. Core Inputs
    # FilePath automatically verifies that the path is a string/Path, exists, and is a file
    gctx_path: FilePath
    inst_info: pd.DataFrame
    gene_info: Optional[pd.DataFrame] = None    # metadata on the genes
    gene_marker: Optional[Union[str, List[str]]] = None     #  selector for landmark/inferred/best inferred

    # 2. Identifiers
    comp_identifier: str = "pert_iname"
    cell_identifier: str = "cell_id"
    instance_identifier: str = "inst_id"

    pert_types: Union[str, List[str]] = Field(default="trt_cp")

    # 3. Computed State Variables (Populated automatically after initialization)
    gene_rids: list = Field(default_factory=list)   # rid = row, gene data 
    meta_df: pd.DataFrame = Field(default_factory=pd.DataFrame)

    # Declare the timers dictionary here
    timers: Dict[str, Timer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_and_initialize_data(self) -> "LINCSDataLoader":
        """Validates input DataFrames and initializes internal state variables.

        This method runs automatically after Pydantic validates the base types.
        It ensures all required columns exist in the provided DataFrames,
        subsets gene identifiers based on the requested feature space, and
        pre-computes the metadata lookup dictionary for faster querying.

        Returns:
            LINCSDataLoader: The fully validated and initialized class instance.

        Raises:
            ValueError: If required columns are missing from `inst_info`,
                `collision_df`, or `gene_info`.
            ValueError: If `gene_marker` is provided but `gene_info` is not.
        """
        # Validate inst_info columns
        req_inst = [
            self.comp_identifier,
            self.cell_identifier,
            self.instance_identifier,
            "pert_type",
        ]
        missing_inst = [c for c in req_inst if c not in self.inst_info.columns]
        if missing_inst:
            raise ValueError(f"Missing column(s) in inst_info: {missing_inst}")

        # Validate and process genes
        if self.gene_marker is not None and self.gene_info is None:
            raise ValueError(
                "Missing 'gene_info' argument. It is required when 'gene_marker' is provided."
            )

        if self.gene_info is not None:
            req_genes = ["feature_space", "gene_id"]
            missing_genes = [c for c in req_genes if c not in self.gene_info.columns]
            if missing_genes:
                raise ValueError(f"Missing column(s) in gene_info: {missing_genes}")

            if self.gene_marker is not None:
                print(f"Subsetting genes using feature_space: {self.gene_marker}")
                # Coerce to list to avoid repetitive type checking
                markers = (
                    [self.gene_marker]
                    if isinstance(self.gene_marker, str)
                    else self.gene_marker
                )

                filtered_df = self.gene_info[
                    self.gene_info["feature_space"].isin(markers)
                ]
                self.gene_rids = sorted(filtered_df["gene_id"].tolist())

                if not self.gene_rids:
                    print(
                        f"Warning: No gene rids were found. Try using one of the possible feature_space values: {self.gene_info['feature_space'].unique()}"
                    )

        # Initialize metadata grouping
        print("Initializing metadata...")

        # "pert_type" is guaranteed to exist due to the req_inst check above
        if isinstance(self.pert_types, list):
            self.meta_df = self.inst_info[self.inst_info["pert_type"].isin(self.pert_types)].copy()
        else:
            self.meta_df = self.inst_info[self.inst_info["pert_type"] == self.pert_types].copy()

        # Initialize timers
        self.timers["io"] = Timer("io")

        return self
        # Load batch is general

    def _load_batch(
        self, cid_list: list, rid_list: Optional[list] = None
    ) -> tuple[dict, list, Any]:
        """Loads a batch of expression data from the GCTX file into a sparse matrix.

        This is an internal helper method. It parses the GCTX file for the specified
        instances (cids) and genes (rids), converts the data into a SciPy CSR
        sparse matrix for memory efficiency, and aggressively cleans up intermediate
        objects.

        Args:
            cid_list (list): A list of instance IDs (columns in the GCTX file) to load.
            rid_list (Optional[list], optional): A list of gene IDs (rows) to load.
                If None, defaults to the pre-initialized `self.gene_rids`.

        Returns:
            tuple:
                - dict: A mapping of instance IDs (strings) to their corresponding
                  row index (integer) in the generated sparse matrix.
                - list[str]: Saves the actual order of the genes which is used for mapping 
                  downstream. 
                - scipy.sparse.csr_matrix: The loaded gene expression matrix, or
                  None if an error occurs during parsing.
        """
        with self.timers["io"]:
            try:
                # Get sample -and gene ID's
                with h5py.File(self.gctx_path, "r") as f:
                    gctx_cols = [x.decode("utf-8") for x in f["0/META/COL/id"][:]]
                    gctx_rows = [x.decode("utf-8") for x in f["0/META/ROW/id"][:]]

                # Determine target row IDs (genes)
                target_rids = rid_list if rid_list is not None else self.gene_rids

                # Filter target lists to only keep IDs that exist in the file
                gctx_cols_set = set(gctx_cols)
                gctx_rows_set = set(gctx_rows)

                safe_sample_ids = [cid for cid in cid_list if cid in gctx_cols_set]
                safe_gene_ids = [rid for rid in target_rids if rid in gctx_rows_set] if target_rids else []

                # Parse GCTX
                if target_rids:
                    data_obj = parse(str(self.gctx_path), cid=safe_sample_ids, rid=safe_gene_ids)
                else:
                    data_obj = parse(str(self.gctx_path), cid=safe_sample_ids)

                gene_order = [str(x) for x in data_obj.data_df.index]
                
                # Convert to df
                temp_df = data_obj.data_df.T

                # Clean up the raw object early to save memory
                del data_obj

                # Create ID map
                inst_id_map = {
                    rid: i for i, rid in enumerate(temp_df.index.astype(str))
                }

                # Sparse Conversion
                expression_matrix = sparse.csr_matrix(temp_df.values)

                # Aggressive Cleanup
                del temp_df

                return inst_id_map, gene_order, expression_matrix

            except Exception as e:
                print(f"Error loading batch due to error: {e}")
                return {}, [], None

    def resolve_gene_rids(self, gene_filters: Dict[str, Union[Any, List[Any]]]) -> list:
        """Filters the gene metadata to extract a targeted list of gene IDs.

        Queries the `gene_info` DataFrame using specific selectors to generate
        a dynamic list of row IDs (rids) for GCTX parsing.

        Args:
            gene_filters (Dict[str, Union[Any, List[Any]]]): A dictionary where
                keys are valid column names (e.g., 'gene_symbol', 'gene_type')
                and values are the specific traits to filter by (scalar or list).

        Returns:
            list: A sorted list of string gene IDs that match the filter criteria.
            Returns an empty list if no genes match. If `gene_info` is not loaded,
            it falls back to returning the default `self.gene_rids`.
        """
        if self.gene_info is None:
            print(
                "Warning: 'gene_info' is not loaded. Cannot filter genes dynamically."
            )
            return self.gene_rids

        query_df = self.gene_info.copy()
        valid_selectors = [
            "gene_symbol",
            "ensembl_id",
            "gene_type",
            "feature_space",
            "gene_id",
        ]

        for col, value in gene_filters.items():
            if col not in valid_selectors:
                print(
                    f"Warning: '{col}' is not a valid gene selector. Valid options are {valid_selectors}. Skipping."
                )
                continue

            if col not in query_df.columns:
                print(
                    f"Warning: Column '{col}' not found in gene_info metadata. Skipping."
                )
                continue

            # Coerce scalar values to lists for uniform .isin() usage
            if not isinstance(value, (list, tuple, set)):
                value = [value]

            query_df = query_df[query_df[col].isin(value)]

        if query_df.empty:
            print(
                "Warning: No genes matched the provided filters. Returning empty list."
            )
            return []

        return sorted(query_df["gene_id"].tolist())

    def get_gene_expression(
        self,
        inst_filters: Dict[str, Union[Any, List[Any]]],
        gene_filters: Optional[Dict[str, Union[Any, List[Any]]]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Retrieves and aligns instance metadata with its corresponding expression data.

        This is the primary public interface for querying the dataset. It filters
        the metadata based on the provided instance criteria, resolves any dynamic
        gene filters, loads the slice of data from the GCTX file, and guarantees
        that the rows of the returned metadata perfectly align with the rows
        of the returned sparse matrix.

        Args:
            inst_filters (Dict[str, Union[Any, List[Any]]]): Criteria to filter
                the instances. Keys must match columns in `inst_info`
                (e.g., {"pert_time": 24, "cell_iname": ["MCF7", "A549"]}).
            gene_filters (Optional[Dict[str, Union[Any, List[Any]]]], optional):
                Criteria to filter the genes. Keys must match valid selectors in
                `gene_info`. Defaults to None (uses class-level gene setup).

        Returns:
            tuple:
                - pd.DataFrame: The filtered instance metadata, sorted so its index rigorously
                  matches the row order of the expression matrix.
                - pd.DataFrame: THe filtered gene metadata, sorted so its index rigorously
                  matches the row order of the expression matrix.
                - scipy.sparse.csr_matrix: The aligned sparse expression matrix.
                  Returns (DataFrame, None) if the query yields empty results
                  or fails to parse.
        """
        query_df = self.meta_df.copy()

        # Filter the metadata
        for col, value in inst_filters.items():
            if col not in query_df.columns:
                print(
                    f"Warning: Column '{col}' not found in metadata. Skipping filter."
                )
                continue

            if not isinstance(value, (list, tuple, set)):
                value = [value]

            query_df = query_df[query_df[col].isin(value)]

        if query_df.empty:
            print("No instances found matching the provided filters.")
            return query_df, None

        # Extract the instance IDs (cids) for the GCTX parser
        cid_list = query_df[self.instance_identifier].tolist()

        # Extract gene ID's (rids) for GCTX parser
        dynamic_rids = None
        if gene_filters:
            dynamic_rids = self.resolve_gene_rids(gene_filters)
            if not dynamic_rids:
                print("Aborting query: Gene filters resulted in 0 genes.")
                return query_df, None

        # Load the data 
        inst_id_map, actual_gene_order, expression_matrix = self._load_batch(cid_list, rid_list=dynamic_rids)

        if expression_matrix is None:
            print("Warning: Expression matrix failed to load.")
            return query_df, pd.DataFrame(), None

        # Reorganize metadata to match expression matrix after loading

        sampleID_to_XID_map_df = pd.DataFrame.from_dict(inst_id_map, orient="index", columns=["expression_matrix_index"])

        query_df = query_df.set_index(self.instance_identifier)
        query_df = query_df.join(sampleID_to_XID_map_df, how="inner") # Dropping non-matches instantly

        query_df = query_df.sort_values("expression_matrix_index").drop(columns=["expression_matrix_index"])
        query_df = query_df.reset_index(names=self.instance_identifier)

        gene_df = pd.DataFrame()

        if self.gene_info is not None:
            gene_df = self.gene_info.query("gene_id in @actual_gene_order").copy()
            
            gene_df = gene_df.set_index("gene_id")
            gene_df = gene_df.reindex(actual_gene_order)
            
            gene_df = gene_df.reset_index()

        return query_df, gene_df, expression_matrix
    
    def create_anndata(self, 
                       verbose: bool,
                       comp_info_merged: pd.DataFrame,
                       inst_filters: Dict[str, Union[Any, List[Any]]],
                       gene_filters: Optional[Dict[str, Union[Any, List[Any]]]] = None) -> ad.AnnData:

                       
        """
        Loads expression data and filters the instance -and gene metadata with the use of get_gene_expression.
        These are then used to build the anndata object. For the observables dataframe, the smiles string in comp_info_merged
        were merged with the instance metadata via pert_id. 

        This function assumes that the only information needed from comp_info_merged for downstream use 
        are the canonical_smiles. If this is not the case, add more attributes to comp_info_smiles and check 
        if any unique pert_id maps to more than one unique attribute value.

        Args:
            verbose: Provides additional information on how many duplicate pert_id's have been dropped
                from comp_info_smiles if set to true.
            comp_info_merged (pd.DataFrame): Compound metadata. It provides the smiles string to map
                to the pert_id's. 
            inst_filters (Dict[str, Union[Any, List[Any]]]): Criteria to filter
                the instance metadata. Keys must match columns in `inst_info`
                (e.g., {"pert_time": 24, "cell_iname": ["MCF7", "A549"]}).
            gene_filters (Optional[Dict[str, Union[Any, List[Any]]]], optional):
                Criteria to filter the genes. Keys must match valid selectors in
                `gene_info`. Defaults to None (uses class-level gene setup).

        Returns:
            adata: ad.AnnData
                Returns the Anndata object of the loaded data
        """

        # Load expression data and filter the instance -and gene metadata
        filtered_inst_metadata, filtered_gene_metadata, X_data = self.get_gene_expression(
            inst_filters=inst_filters, 
            gene_filters=gene_filters
            )

        # Build the observables Dataframe

        # 1. Subset comp_info_merge to contain pert_id's and their corresponding smiles
        comp_info_smiles = comp_info_merged[["pert_id", "canonical_smiles", "fingerprint_smiles"]]

        # 2. If there are multiple pert_id's, deduplicate
        if comp_info_smiles.duplicated(subset="pert_id").any():

            # Verify that no unique pert_id maps to different smiles strings/fingerprints. Mapping won't work otherwise.
            smiles_per_id = comp_info_smiles.groupby("pert_id")["canonical_smiles"].nunique()

            if smiles_per_id.max() > 1:
                warnings.warn(f"Data Integrity Warning: Found different SMILES strings for the same pert_id.")
            
            fingerprint_per_id = comp_info_smiles.groupby("pert_id")["fingerprint_smiles"].nunique()

            if fingerprint_per_id.max() > 1:
                warnings.warn(f"Data Integrity Warning: Found different Fingerprints for the same pert_id.")
            

            rows_before = len(comp_info_merged)
            comp_info_smiles = comp_info_smiles.drop_duplicates(subset="pert_id", keep="first")

            if verbose:
                rows_after = len(comp_info_merged)
                print(f"A number of {rows_before-rows_after} rows have been dropped due to duplicate pert_id's.")


        
        # 3. Merge the smiles strings from comp_info_smiles with the instance metadata to form the observables
        
        #control = ["ctl_x", "ctl_vehicle", "ctl_untrt", "ctl_vector"] 

        is_control_data = filtered_inst_metadata[filtered_inst_metadata['control'] == 1]
        if not is_control_data.empty:
            # Use left merge to keep control data, even if they don't have smiles 
            merged_obs_df = pd.merge(left=filtered_inst_metadata, right=comp_info_smiles, how="left", on="pert_id")
        else:
            # Use inner merge to drop profiles that actually lack smiles strings. (Due to comp_info_merged not storing them)
            merged_obs_df = pd.merge(left=filtered_inst_metadata, right=comp_info_smiles, how="inner", on="pert_id")
            
            # Realign expression data to obs after dropping rows from obs due to inner merge
            X_data = X_data[merged_obs_df.index, :]
            
            # Reset the metadata index after it got fragmented from inner merge
            merged_obs_df = merged_obs_df.reset_index(drop=True)
        
        #merged_obs_df = pd.merge(left=filtered_inst_metadata, right=comp_info_smiles, how="left", on="pert_id")

        # Build adata object 
        merged_obs_df = merged_obs_df.set_index(self.instance_identifier) # the index is now the sample_id
        filtered_gene_metadata = filtered_gene_metadata.set_index("gene_id")  

        adata = ad.AnnData(X=X_data, obs=merged_obs_df, var=filtered_gene_metadata)

        print(f"Adata object created")

        return adata

    def print_performance(self, restart: bool = False) -> None:
        """Prints the accumulated execution time for all registered timers.

        Iterates through the internal `timers` dictionary and outputs a formatted
        performance report to the console. Useful for profiling I/O operations
        and identifying bottlenecks in the data loading pipeline.

        Args:
            restart (bool, optional): If True, resets all timer accumulators
                back to zero immediately after printing their current values.
                Defaults to False.

        Returns:
            None
        """
        print("----- Performance -----")
        for key, timer in self.timers.items():
            if restart:
                timer.print_restart()
            else:
                print(timer)
        print("--------------------------\n")


if __name__ == "__main__":
    task = sys.argv[1]

    if task == "load_comp":
        metadata_file = Path(sys.argv[2]) 

        comp_meta = load_metadata_txt(metadata_file)
        comp_meta.to_parquet("comp_metadata.parquet", index=False)

    elif task == "load_inst":
        metadata_file = Path(sys.argv[2])  

        inst_meta = load_metadata_txt(metadata_file)
        inst_meta.to_parquet("inst_metadata.parquet", index=False)

    elif task == "process_expression":

        # Input ----
        comp_file = Path(sys.argv[2])  
        inst_file = Path(sys.argv[3])   
        gene_file = Path(sys.argv[4])   
        gctx_cp   = Path(sys.argv[5])   
        gctx_ctl  = Path(sys.argv[6])   
        
        df_comp = pd.read_parquet(comp_file)
        df_inst = pd.read_parquet(inst_file)
        
        df_gene = load_metadata_txt(gene_file)

        # Loading GE Data ----
        print("Compiling AnnData via LINCSDataLoader...")
        lincs_adata = load_ge(
            inst_metadata=df_inst, 
            comp_metadata=df_comp, 
            gene_metadata=df_gene, 
            gctx_cp_path=str(gctx_cp), 
            gctx_ctl_path=str(gctx_ctl)
        )

        # Check size of loaded and filtered expression data ----

        matrix_bytes = lincs_adata.X.nbytes if hasattr(lincs_adata.X, 'nbytes') else 0
        if matrix_bytes == 0 and hasattr(lincs_adata.X, 'data'): # If it's a sparse matrix
            matrix_bytes = lincs_adata.X.data.nbytes + lincs_adata.X.indices.nbytes + lincs_adata.X.indptr.nbytes
            
        obs_bytes = lincs_adata.obs.memory_usage(deep=True).sum()
        var_bytes = lincs_adata.var.memory_usage(deep=True).sum()
        
        total_gigabytes = (matrix_bytes + obs_bytes + var_bytes) / (1024 ** 3)
        
        print("\n" + "="*50)
        print(f"DRY RUN COMPLETE: Preprocessing and filtering successfully finished.")
        print(f"Estimated size of the final compiled AnnData object: {total_gigabytes:.2f} GB")
        print("="*50 + "\n")
        
        # Output ----
        lincs_adata.write("loaded_dataset.h5ad", compression="gzip")

        del lincs_adata # delete from memory to save space
        
    else:
        print(f"Error: Unknown task '{task}'", file=sys.stderr)
        sys.exit(1)