# Imports -----

import pandas as pd
import time

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
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

# Load GE from LINCS datasets ----

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
    gene_info: Optional[pd.DataFrame] = None
    gene_marker: Optional[Union[str, List[str]]] = None

    # 2. Identifiers
    comp_identifier: str = "pert_iname"
    cell_identifier: str = "cell_id"
    instance_identifier: str = "inst_id"

    # 3. Computed State Variables (Populated automatically after initialization)
    gene_rids: list = Field(default_factory=list)
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
        self.meta_df = self.inst_info[self.inst_info["pert_type"] == "trt_cp"].copy()
        # Initialize timers
        self.timers["io"] = Timer("io")

        return self
        # Load batch is general

    def _load_batch(
        self, cid_list: list, rid_list: Optional[list] = None
    ) -> tuple[dict, Any]:
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
                - scipy.sparse.csr_matrix: The loaded gene expression matrix, or
                  None if an error occurs during parsing.
        """
        with self.timers["io"]:
            try:
                # 1. Determine target row IDs (genes)
                # Prioritize the dynamically passed rid_list; fallback to class-level gene_rids
                target_rids = rid_list if rid_list is not None else self.gene_rids

                # 2. Parse GCTX
                if target_rids:
                    data_obj = parse(str(self.gctx_path), cid=cid_list, rid=target_rids)
                else:
                    data_obj = parse(str(self.gctx_path), cid=cid_list)

                # 3. To DataFrame
                temp_df = data_obj.data_df.T

                # Clean up the raw object early to save memory
                del data_obj

                # 4. Create ID map
                inst_id_map = {
                    rid: i for i, rid in enumerate(temp_df.index.astype(str))
                }

                # 5. Sparse Conversion
                expression_mat = sparse.csr_matrix(temp_df.values)

                # 6. Aggressive Cleanup
                del temp_df

                return inst_id_map, expression_mat

            except Exception as e:
                print(f"Error loading batch due to error: {e}")
                return {}, None

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
    ) -> tuple[pd.DataFrame, Any]:
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
                - pd.DataFrame: The filtered metadata, sorted so its index rigorously
                  matches the row order of the expression matrix.
                - scipy.sparse.csr_matrix: The aligned sparse expression matrix.
                  Returns (DataFrame, None) if the query yields empty results
                  or fails to parse.
        """
        # 1. Start with all valid metadata
        query_df = self.meta_df.copy()

        # 2. Iteratively apply filters
        for col, value in inst_filters.items():
            if col not in query_df.columns:
                print(
                    f"Warning: Column '{col}' not found in metadata. Skipping filter."
                )
                continue

            # Coerce scalar values to lists for uniform .isin() usage
            if not isinstance(value, (list, tuple, set)):
                value = [value]

            query_df = query_df[query_df[col].isin(value)]

        # 3. Handle empty queries
        if query_df.empty:
            print("No instances found matching the provided filters.")
            # Return empty DataFrame and None to prevent downstream crashes
            return query_df, None

        # 4. Extract the instance IDs (cids) for the GCTX parser
        cid_list = query_df[self.instance_identifier].tolist()

        # Resolve dynamic genes if filters are provided
        dynamic_rids = None
        if gene_filters:
            dynamic_rids = self.resolve_gene_rids(gene_filters)
            if not dynamic_rids:
                print("Aborting query: Gene filters resulted in 0 genes.")
                return query_df, None

        # Load the data passing BOTH lists
        inst_id_map, expression_mat = self._load_batch(cid_list, rid_list=dynamic_rids)

        if expression_mat is None:
            print("Warning: Expression matrix failed to load.")
            return query_df, None

        # 6. Strict Matrix Alignment
        # Map the true matrix row index back to the metadata DataFrame
        query_df["_matrix_idx"] = query_df[self.instance_identifier].map(inst_id_map)

        # Drop any instances that the parser failed to find in the GCTX
        query_df = query_df.dropna(subset=["_matrix_idx"])

        # Sort the metadata so its row order perfectly matches the sparse matrix row order
        query_df = query_df.sort_values("_matrix_idx").drop(columns=["_matrix_idx"])

        # Reset the index for clean downstream usage
        query_df = query_df.reset_index(drop=True)

        return query_df, expression_mat

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