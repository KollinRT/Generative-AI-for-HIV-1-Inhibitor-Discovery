"""
Code utilized to generate Morgan Fingerprints for Round 2 analysis.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import selfies as sf


# Helper function to convert SELFIES to RDKit Mol
def selfies_to_mol(selfies_string):
    """
    Args:
        selfies_string: str
            SELFIES string to convert to mol
    Returns:
        rdkit.Chem.MolFromSmiles object
    """
    smiles = sf.decoder(str(selfies_string))  # Decode SELFIES to SMILES
    return Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule


def make_fingerprint_thisthat(df):
    """
    Args:
        df : pd.DataFrame
            A Pandas Dataframe object that contains the SELFIES strings.
    Returns:
        df : pd.DataFrame
            A Pandas Dataframe object that contains the Fingerprints.
    """
    # Load dataset
    # df = pd.read_csv(f"/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/data/molecule_data_model_{key}.csv")
    # df = pd.read_csv(f"/home/trujillok/Desktop/Thesis/WIP_Thesis/data/molecule_data_model_nada.csv")

    # Convert SELFIES to RDKit Molecule
    df["Molecule"] = df["selfies"].apply(selfies_to_mol)

    # Remove invalid molecules
    df = df[df["Molecule"].notnull()]

    # Define fingerprint parameters
    fp_radius = 2  # Morgan radius (ECFP4 uses diameter=4 → radius=2)
    fp_size = 2048  # Standard fingerprint size

    # Convert fingerprints to NumPy arrays
    fingerprints = []
    for mol in df["Molecule"]:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_size)
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)  # Efficiently convert to NumPy
        fingerprints.append(arr)

    # Drop df.molecules
    # df = df.drop(columns=['Molecule'])
    # df = df.drop(columns=['MW', 'numC', 'chain_length', 'cLogP', 'numRings'])

    # Add fingerprints to df
    df["Fingerprint"] = fingerprints
    df["Fingerprint"] = df["Fingerprint"].apply(
        lambda arr: "".join(str(x) for x in arr)
    )  # Convert to string

    # Stack fingerprints into a single NumPy array
    fingerprint_matrix = np.vstack(fingerprints)

    return df
