# Imports

import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List, Optional, Union, Literal, Annotated
from pydantic import ConfigDict, Field, StrictStr, validate_call

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, inchi, rdFMCS, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from skfp.fingerprints import (
    MAPFingerprint,
    ERGFingerprint,
    PhysiochemicalPropertiesFingerprint,
    TopologicalTorsionFingerprint,
    SECFPFingerprint
)
_worker_enumerator = None


# Get fingerprints ----

def get_fingerprint(
    representation: Union[StrictStr, None],
    input_type: Literal["smiles", "inchi"] = "smiles",
    num_bits: int = 1024, # 2. Replaced Field() with standard Python default
) -> Union[np.ndarray, None]:
    """
    Computes a molecular fingerprint from a chemical representation.
    
    Args:
        representation: The chemical string (SMILES or InChI).
        input_type: The format of the input string.
        num_bits: Length of the fingerprint bitstring (typically a power of 2).
    """
    global _worker_enumerator

    # 2. Initialize ONLY ONCE per worker process (e.g., 8 times total, not 20k)
    if _worker_enumerator is None:
        _worker_enumerator = (
            rdMolStandardize.TautomerEnumerator()
        )  # To unify the tautomers

    # Handle empty inputs
    if not representation:
        return None

    mol = None
    try:
        if input_type == "smiles":
            mol = Chem.MolFromSmiles(representation)
        elif input_type == "inchi":
            mol = inchi.MolFromInchi(representation)
    except Exception:
        return None

    # Check if mol is none
    if mol is None:
        return None

    try:
        # Use the cached enumerator
        mol = _worker_enumerator.Canonicalize(mol)

        # Canonicalize might return None if it fails
        if mol is None:
            return None

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, useFeatures=True, nBits=num_bits
        ).ToBitString()
        return fingerprint

    except Exception as e:
        print(f"Error processing {representation}: {e}")
        # Catch ALL errors and return None.
        return None
    
def get_scaffold_generic(mol: Chem.rdchem.Mol) -> Union[StrictStr, None]:
    """
    Converts RDKit mol to a generic all-carbon, single-bond scaffold.
    Returns the SMILES string or None if conversion fails.
    """
    try:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None

def get_skeleton_smiles(mol: Chem.rdchem.Mol) -> StrictStr:
    """
    Transforms all atoms to neutral carbons and returns the SMILES.
    """
    m_copy = Chem.RWMol(mol)
    for atom in m_copy.GetAtoms():
        atom.SetAtomicNum(6)      # Turn to Carbon
        atom.SetFormalCharge(0)   # Remove charge
        atom.SetIsAromatic(False) # Remove aromatic flags
        
    return Chem.MolToSmiles(m_copy, isomericSmiles=False)

# Validations ----

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def are_molecules_equivalent(
    smiles1: StrictStr, 
    smiles2: StrictStr
) -> bool:
    """
    Compares two SMILES strings to see if they represent the same molecule.
    Accounts for: Numbering, atom ordering, and stereochemistry.
    """

    # Convert to RDKit Mol objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Check if SMILES were valid RDKit strings
    if mol1 is None or mol2 is None:
        return False

    # Canonicalize and compare
    # isomericSmiles=True captures stereochemical differences (chiral centers, double bonds)
    can_s1 = Chem.MolToSmiles(mol1, isomericSmiles=True)
    can_s2 = Chem.MolToSmiles(mol2, isomericSmiles=True)

    return can_s1 == can_s2


SmilesList = Annotated[List[str], Field(min_length=1)]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def draw_molecules(
    smiles: SmilesList,
    legends: Optional[List[str]] = None
) -> Union[Draw.MolDraw2D, object]: # Returns an RDKit Canvas/Image object
    """
    Parses SMILES, aligns them using MCS for visual consistency, and returns a grid image.
    """
    # 1. Parsing
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [mol for mol in mols if mol is not None]

    if len(mols) != len(smiles):
        print(
            f"Warning! There were {len(smiles) - len(mols)} None molecules. Legend will not work!"
        )
        legends = None
    if legends is None or len(legends) != len(smiles):
        legends = [f"Mol {i + 1}" for i in range(len(smiles))]

    # 2. Alignment Logic (MCS)
    if len(mols) > 1:
        # OPTIMIZATION: Added a timeout (in seconds).
        # Complex molecules with 'CompareAny' can take forever to align.
        # 1 second is usually enough for a visual alignment; if it takes longer, skip it.
        mcs = rdFMCS.FindMCS(
            mols,
            completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareAny,
            timeout=1,
        )

        aligned = False

        # Check if MCS found something and wasn't canceled by timeout
        if mcs.numAtoms > 0 and not mcs.canceled:
            try:
                core = Chem.MolFromSmarts(mcs.smartsString)
                core.UpdatePropertyCache()
                AllChem.Compute2DCoords(core)

                for m in mols:
                    try:
                        AllChem.GenerateDepictionMatching2DStructure(m, core)
                    except (ValueError, Chem.AtomValenceException):
                        AllChem.Compute2DCoords(m)

                aligned = True
            except (Chem.AtomValenceException, ValueError, Exception):
                pass

        # If alignment failed, timed out, or threw an error, generate standard coords
        if not aligned:
            for m in mols:
                AllChem.Compute2DCoords(m)
    else:
        if mols:
            AllChem.Compute2DCoords(mols[0])

    # 3. Dynamic Layout Settings
    n_mols = len(mols)
    mols_per_row = n_mols if n_mols < 3 else 3

    # 4. Draw
    dopts = Draw.MolDrawOptions()
    dopts.addStereoAnnotation = True
    dopts.fixedBondLength = 35

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(450, 450),
        legends=legends,
        drawOptions=dopts,
    )

    return img

# Analyze reasons for collision ----

def analyze_fingerprint_collision(
    smiles_list: List[str] = Field(..., min_length=1)
) -> List[str]:
    reasons = set()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    if any(m is None for m in mols):
        return ["Invalid SMILES"]

    # --- Data Extraction ---
    formulas = [rdMolDescriptors.CalcMolFormula(m) for m in mols]
    skeletons = [get_scaffold_generic(m) for m in mols]
    # Map non-isomeric SMILES to their various isomeric versions found in the group
    connectivity_map = defaultdict(set)
    for m in mols:
        non_iso = Chem.MolToSmiles(m, isomericSmiles=False)
        iso = Chem.MolToSmiles(m, isomericSmiles=True)
        connectivity_map[non_iso].add(iso)

    # 5. Mixture Check
    if len(set(s.count(".") for s in smiles_list)) > 1: 
        reasons.add("Dimer/Salt Variation")

    # 1. Composition/Analog Check
    if len(set(formulas)) > 1:
        if len(smiles_list) > 2:
            reasons.add("Compositional Isomer (Different Formula)")
        elif len(smiles_list) == 2 and "Dimer/Salt Variation" not in reasons:
            reasons.add("Compositional Isomer (Different Formula)")

    # 2. Structural/Positional Isomer Check
    # Same formula, but different non-isomeric connectivity
    if len(set(formulas)) == 1 and len(connectivity_map) > 1:
        reasons.add("Structural Isomers (Positional)")

    # 3. Bioisostere Check
    # If they aren't identical connectivity, but share a skeleton
    if len(connectivity_map) > 1 and len(set(skeletons)) == 1:
        reasons.add("Bioisosteres (Same Skeleton)")

    # 4. Stereochemical Analysis (Per Connectivity Group)
    found_r_s_conflict = False
    found_specificity_issue = False

    for iso_variants in connectivity_map.values():
        if len(iso_variants) > 1:
            # Check if any have @ and some don't
            has_stereo = [("@" in s or "/" in s or "\\" in s) for s in iso_variants]

            # If some are defined and some are not
            if any(has_stereo) and not all(has_stereo):
                found_specificity_issue = True

            # If we have at least two different defined stereoisomers
            # (e.g., both have @ but they are different strings)
            defined_variants = [
                s for s in iso_variants if "@" in s or "/" in s or "\\" in s
            ]  # List of smiles that have some stereo indicators
            if len(set(defined_variants)) > 1:
                found_r_s_conflict = True

    if found_r_s_conflict:
        reasons.add("Stereoisomers (R/S conflict)")
    if found_specificity_issue:
        reasons.add("Defined vs Undefined Stereo")

    # 6. Fallback for True Hash Collision
    if not reasons and len(set(Chem.MolToSmiles(m) for m in mols)) > 1:
        reasons.add("True Hash Collision (Unclassified)")
    # 7. Last possible explanation = True Duplicates
    if not reasons:
        # Generate the canonical isomeric SMILES for all molecules in the group
        # This is the exact same logic inside your are_molecules_equivalent function
        canonical_smiles = set(Chem.MolToSmiles(m, isomericSmiles=True) for m in mols)
        
        # If they all collapse to a single unique canonical string, they are identical
        if len(canonical_smiles) == 1:
            reasons.add("True duplicates")
        else:
            reasons.add("Error in the labelling")
    return sorted(list(reasons))

def inspect_reason(
    df: pd.DataFrame, 
    reason: str, 
    max_print: int, 
    draw_mols: bool = False, 
    seed: int = 2025
):
    """
    Samples and prints compound information for a specific collision reason.
    """
    # Direct boolean filtering is faster than a full groupby if we only need one slice
    group = df[df["Reasons"] == reason]
    
    # Safety check
    if group.empty:
        print(f"#### {reason} ####\nCategory not found.\n")
        return
        
    print(f"#### {reason} ####\nNo. fingerprints: {len(group)}")
    
    # random_state locally scopes the seed for this specific operation
    sampled_group = group.sample(n=min(max_print, len(group)), random_state=seed)
    
    for _, row in sampled_group.iterrows():
        identifiers = [
            f"{sm}: {iname}, {pert_id}"
            for sm, iname, pert_id in zip(row["Smiles"], row["cmap_name"], row["pert_id"])
        ]
        print("\n".join(identifiers))
        
        if draw_mols:
            display(draw_molecules(row["Smiles"]))
        else:
            print(30 * "-")
    print(100 * "-")

# for tetsing differet fp methods

from collections import defaultdict
import pandas as pd
import numpy as np



# 1. Initialize fingerprint transformers from skfp
fingerprint_dict = {
    # include_chirality=True turns MAP4 into MAP4C (stereochemistry-aware)
    "MAP4": MAPFingerprint(fp_size=1024, radius=2, include_chirality=False),
    "MAP4C": MAPFingerprint(fp_size=1024, radius=2, include_chirality=True),
    "erg": ERGFingerprint(),
    "Physicochemical": PhysiochemicalPropertiesFingerprint(),
    "Topological Torsion": TopologicalTorsionFingerprint(fp_size=1024),
    "SECFP Fingerprint": SECFPFingerprint(fp_size=1024)
}


# Registry of scikit-fingerprints transformers
FP_TRANSFORMERS = {
    # MAP4 paper implementation (set include_chirality=True for MAP4C)
    "map4": MAPFingerprint(fp_size=1024, radius=2, include_chirality=False),
    "map4c": MAPFingerprint(fp_size=1024, radius=2, include_chirality=True),
    "erg": ERGFingerprint(),
    "physicochemical": PhysiochemicalPropertiesFingerprint(),
    "topological_torsion": TopologicalTorsionFingerprint(fp_size=1024),
    "SECFP": SECFPFingerprint(fp_size=1024)
}

def get_fingerprint_all(
    representation: Union[str, List[str]],
    fp_type: Literal[
        "morgan",
        "map4",
        "map4c",
        "erg",
        "physicochemical",
        "topological_torsion",
        "SECFP"
    ] = "morgan",
    input_type: Literal["smiles", "inchi"] = "smiles",
    num_bits: int = 1024,
) -> Union[str, List[str], None]:
    """
    Computes fingerprint bitstrings for single or multiple SMILES/InChI inputs.
    Delegates to get_fingerprint for 'morgan', and FP_TRANSFORMERS (skfp) for others.
    """
    is_single_input = isinstance(representation, str)
    reps = [representation] if is_single_input else representation

    if not reps:
        return None

    # --- 1. Call custom get_fingerprint for Morgan ---
    if fp_type == "morgan":
        bitstrings = [
            get_fingerprint(r, input_type=input_type, num_bits=num_bits)
            for r in reps
        ]
        return bitstrings[0] if is_single_input else bitstrings

    # --- 2. Call scikit-fingerprints transformers for other types ---
    elif fp_type in FP_TRANSFORMERS:
        try:
            transformer = FP_TRANSFORMERS[fp_type]
            fp_matrix = transformer.transform(reps)
            bitstrings = ["".join(row.astype(str)) for row in fp_matrix]
            return bitstrings[0] if is_single_input else bitstrings
        except Exception as e:
            print(f"Error generating {fp_type} fingerprint: {e}")
            return None
    else:
        valid_types = ["morgan"] + list(FP_TRANSFORMERS.keys())
        raise ValueError(
            f"Unknown fp_type '{fp_type}'. Valid options: {valid_types}"
        )
    
def analyze_fingerprint_collision_all(smiles_list: List[str]) -> List[str]:
    """
    Analyzes a set of SMILES strings that resulted in identical fingerprints
    and categorizes the structural reason for the collision.
    """
    if not smiles_list or len(smiles_list) < 2:
        return []

    reasons = set()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Filter invalid SMILES
    if any(m is None for m in mols):
        return ["Invalid SMILES"]

    # --- Feature Extraction ---
    formulas = [rdMolDescriptors.CalcMolFormula(m) for m in mols]
    skeletons = [get_scaffold_generic(m) for m in mols]
    
    # Map non-isomeric SMILES to their isomeric versions within the group
    connectivity_map = defaultdict(set)
    for m in mols:
        non_iso = Chem.MolToSmiles(m, isomericSmiles=False)
        iso = Chem.MolToSmiles(m, isomericSmiles=True)
        connectivity_map[non_iso].add(iso)

    # 1. Salt/Dimer Variations
    if len(set(s.count(".") for s in smiles_list)) > 1:
        reasons.add("Dimer/Salt Variation")

    # 2. Compositional Isomers (Different Molecular Formulas)
    if len(set(formulas)) > 1:
        reasons.add("Compositional Isomer (Different Formula)")

    # 3. Positional / Structural Isomers
    if len(set(formulas)) == 1 and len(connectivity_map) > 1:
        reasons.add("Structural Isomers (Positional)")

    # 4. Bioisosteres
    if len(connectivity_map) > 1 and len(set(skeletons)) == 1:
        reasons.add("Bioisosteres (Same Skeleton)")

    # 5. Stereochemistry Analysis
    found_r_s_conflict = False
    found_specificity_issue = False

    for iso_variants in connectivity_map.values():
        if len(iso_variants) > 1:
            has_stereo = [("@" in s or "/" in s or "\\" in s) for s in iso_variants]

            if any(has_stereo) and not all(has_stereo):
                found_specificity_issue = True

            defined_variants = [
                s for s in iso_variants if "@" in s or "/" in s or "\\" in s
            ]
            if len(set(defined_variants)) > 1:
                found_r_s_conflict = True

    if found_r_s_conflict:
        reasons.add("Stereoisomers (R/S conflict)")
    if found_specificity_issue:
        reasons.add("Defined vs Undefined Stereo")

    # 6. Fallback Checks
    if not reasons:
        canonical_smiles = set(Chem.MolToSmiles(m, isomericSmiles=True) for m in mols)
        if len(canonical_smiles) == 1:
            reasons.add("True duplicates")
        elif len(set(Chem.MolToSmiles(m, isomericSmiles=False) for m in mols)) > 1:
            reasons.add("True Hash Collision (Unclassified)")
        else:
            reasons.add("Error in the labelling")

    return sorted(list(reasons))