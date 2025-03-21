# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# """
# Code from https://github.com/UnixJunkie/mol2ecfp4/blob/master/ecfp4.py
# """
# from rdkit.Chem.Scaffolds import MurckoScaffold
# import selfies as sf
#
# def selfies_to_mol(selfies_string):
#     """Convert a SELFIES string to an RDKit Mol object."""
#     smiles = sf.decoder(selfies_string)  # Decode SELFIES to SMILES
#     return Chem.MolFromSmiles(smiles)    # Convert SMILES to RDKit molecule
#
# def extract_scaffold_from_selfies(selfies_string):
#     """Extract the Bemis-Murcko scaffold from a SELFIES string."""
#     mol = selfies_to_mol(selfies_string)
#     if mol is None:
#         return None
#     scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#     return Chem.MolToSmiles(scaffold)
#
# # ECFP4 is used in ZINC15
# # In ECFP4, 4 stands for the diameter of the atom environment
# fp_diameter = 4
# # but rdkit wants a radius
# fp_radius = int(fp_diameter / 2)
#
# # fp = AllChem.GetMorganFingerprint(mol, fp_radius)
#
# # Load the dataset
# # df = pd.read_csv("/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/molecules.csv")
# # df = pd.read_csv("/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/data/molecule_data_model_nada.csv")
# df = pd.read_csv("/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/data/molecule_data_model_lipinski.csv")
#
# # # Assuming the column containing SMILES is named 'SMILES'
# # if 'SMILES' not in df.columns:
# #     raise ValueError("Input CSV must contain a column named 'SMILES'.")
# #
# #
# #
# # # Generate molecules from SMILES
# # df['Molecule'] = df['SMILES'].apply(Chem.MolFromSmiles)
# #
# # # Filter out invalid molecules
# # df = df[df['Molecule'].notnull()]
# #
# # # fp = AllChem.GetMorganFingerprint(mol, fp_radius)
# #
# # print(df.head())
# # # for mol in df['Molecule']:
# # #     df['ECFP4'] = df['Molecule'].apply(AllChem.GetMorganFingerprint(mol, fp_radius))
# #
# #
# # # # Apply the function correctly
# # # df['ECFP4'] = df['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprint(mol, fp_radius))
# #
# #
# # # Generate ECFP4 fingerprints
# # # Choose GetMorganFingerprintAsBitVect if you need a dense bit vector representation
# # df['ECFP4'] = df['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius))
# #
# # # Optionally, convert the fingerprint to a list of bits for easier handling
# # df['ECFP4_bits'] = df['ECFP4'].apply(lambda fp: list(fp.ToBitString()))
# #
# # # # Convert scaffolds back to SMILES for easier visualization
# # # df['Scaffold_SMILES'] = df['Scaffold'].apply(lambda x: Chem.MolToSmiles(x) if x else None)
# #
# # print(df.head())
#
# # Generate molecules from SMILES
# # df['smiles'] = df['selfies'].apply(selfies_to_mol)
# # df['Molecule'] = df['smiles'].apply(Chem.MolFromSmiles)
# df['Molecule'] = df['selfies'].apply(selfies_to_mol)
#
#
# # Filter out invalid molecules
# df = df[df['Molecule'].notnull()]
#
# # fp = AllChem.GetMorganFingerprint(mol, fp_radius)
#
# print(df.head())
# # for mol in df['Molecule']:
# #     df['ECFP4'] = df['Molecule'].apply(AllChem.GetMorganFingerprint(mol, fp_radius))
#
#
# # # Apply the function correctly
# # df['ECFP4'] = df['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprint(mol, fp_radius))
#
#
# # Generate ECFP4 fingerprints
# # Choose GetMorganFingerprintAsBitVect if you need a dense bit vector representation
# df['ECFP4'] = df['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius))
#
# # Optionally, convert the fingerprint to a list of bits for easier handling
# df['ECFP4_bits'] = df['ECFP4'].apply(lambda fp: list(fp.ToBitString()))
#
# # # Convert scaffolds back to SMILES for easier visualization
# # df['Scaffold_SMILES'] = df['Scaffold'].apply(lambda x: Chem.MolToSmiles(x) if x else None)
#
# # df.to_csv("./saved_ecfp4_mols_nada.csv")
# output_path = "./saved_ecfp4_mols_lipinski.csv"
# df.to_csv(output_path, index=False)
#
# print(f"Processed data saved to {output_path}")
#
# print(df.head())
#
# from rdkit.ML.Cluster import Butina
# from rdkit.DataStructs import TanimotoSimilarity
# from rdkit import DataStructs
#
#
# # Function to perform Butina clustering
# def butina_clustering(fingerprints, threshold=0.7):
#     """
#     Perform Butina clustering based on Tanimoto similarity.
#
#     Args:
#     - fingerprints: List of RDKit ExplicitBitVect fingerprints.
#     - threshold: Tanimoto similarity threshold for clustering.
#
#     Returns:
#     - clusters: List of clusters, where each cluster is a list of indices.
#     """
#     # Calculate the Tanimoto distances
#     dists = []
#     n_fps = len(fingerprints)
#     for i in range(1, n_fps):
#         sims = [1 - TanimotoSimilarity(fingerprints[i], fingerprints[j]) for j in range(i)]
#         dists.extend(sims)
#
#     # Perform Butina clustering
#     clusters = Butina.ClusterData(dists, n_fps, threshold, isDistData=True)
#     return clusters
#
#
# # Convert the fingerprints to RDKit ExplicitBitVect objects
# fingerprints = list(df['ECFP4'])
#
# # Perform Butina clustering
# threshold = 0.7  # Adjust threshold as needed
# clusters = butina_clustering(fingerprints, 0.7)
#
# # Output the results
# print(f"Number of clusters formed: {len(clusters)}")
#
# # Assign cluster IDs to the molecules in the DataFrame
# df['ClusterID'] = -1  # Initialize with -1
# for cluster_id, cluster in enumerate(clusters):
#     for molecule_index in cluster:
#         df.at[molecule_index, 'ClusterID'] = cluster_id
#
# # Save the clustered data to a CSV file
# output_path_with_clusters = "./saved_ecfp4_mols_with_clusters.csv"
# df.to_csv(output_path_with_clusters, index=False)
#
# print(f"Clustered data saved to {output_path_with_clusters}")
# print(df.head())
#
# # from rdkit import Chem
# # from rdkit.Chem import AllChem
# #
# # # Example: Generate Morgan fingerprint for a molecule
# # mol = Chem.MolFromSmiles('CCO')
# # fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
# #
# # from sklearn.decomposition import PCA
# # reduced_data = PCA(n_components=50).fit_transform(fingerprint_matrix)
# #
# # from rdkit.ML.Cluster import Butina
# # from rdkit.DataStructs import TanimotoSimilarity
# #
# # # Example: Cluster using Tanimoto similarity
# # def cluster_mols(fps, threshold=0.7):
# #     dists = []
# #     for i in range(1, len(fps)):
# #         sims = [1 - TanimotoSimilarity(fps[i], fps[j]) for j in range(i)]
# #         dists.extend(sims)
# #     return Butina.ClusterData(dists, len(fps), threshold, isDistData=True)
# #
# # clusters = cluster_mols(fingerprint_list)
#
#

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.DataStructs import TanimotoSimilarity
import selfies as sf


# Helper function to convert SELFIES to RDKit Mol
def selfies_to_mol(selfies_string):
    smiles = sf.decoder(selfies_string)  # Decode SELFIES to SMILES
    return Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule

# TODO: Convert to fingerprints
def make_fingerprint_thisthat(df):
    """
    @param df:
    # @param key:

    return:
    """
    # Load dataset
    # df = pd.read_csv(f"/media/kollin/WindowsSecondary/ThesisBU/Thesis/WIP_Thesis/data/molecule_data_model_{key}.csv")
    # df = pd.read_csv(f"/home/trujillok/Desktop/Thesis/WIP_Thesis/data/molecule_data_model_nada.csv")

    # Convert SELFIES to RDKit Molecule
    df['Molecule'] = df['selfies'].apply(selfies_to_mol)

    # Remove invalid molecules
    df = df[df['Molecule'].notnull()]

    # Define fingerprint parameters
    fp_radius = 2  # Morgan radius (ECFP4 uses diameter=4 → radius=2)
    fp_size = 2048  # Standard fingerprint size

    # Convert fingerprints to NumPy arrays
    fingerprints = []
    for mol in df['Molecule']:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_size)
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)  # Efficiently convert to NumPy
        fingerprints.append(arr)

    # Drop df.molecules
    # df = df.drop(columns=['Molecule'])
    # df = df.drop(columns=['MW', 'numC', 'chain_length', 'cLogP', 'numRings'])

    # Add fingerprints to df
    df['Fingerprint'] = fingerprints

    print("columns in thisthat")
    print(df.columns)

    # Stack fingerprints into a single NumPy array
    fingerprint_matrix = np.vstack(fingerprints)
    print(f"Shape of fingerprint matrix: {fingerprint_matrix.shape}")

    # TODO: NEW this doesn't give the Clusters...

    return df


# # Define function for Butina clustering with batch processing
# def butina_clustering(fingerprints, threshold=0.7, batch_size=50000):
#     """
#     Perform Butina clustering using batch processing.
#
#     Args:
#     - fingerprints: NumPy array of Morgan fingerprints.
#     - threshold: Tanimoto similarity threshold for clustering.
#     - batch_size: Number of molecules processed per batch.
#
#     Returns:
#     - List of clusters, where each cluster is a list of indices.
#     """
#     n_fps = len(fingerprints)
#     clusters = []
#
#     for start_idx in range(0, n_fps, batch_size):
#         end_idx = min(start_idx + batch_size, n_fps)
#         dists = []
#
#         # Compute Tanimoto similarity for the batch
#         for i in range(start_idx + 1, end_idx):
#             sims = [1 - TanimotoSimilarity(DataStructs.CreateFromBitString("".join(map(str, fingerprints[i]))),
#                                            DataStructs.CreateFromBitString("".join(map(str, fingerprints[j]))))
#                     for j in range(start_idx, i)]
#             dists.extend(sims)
#
#         # Cluster the current batch
#         batch_clusters = Butina.ClusterData(dists, end_idx - start_idx, threshold, isDistData=True)
#         clusters.extend(batch_clusters)
#
#     return clusters


# Perform clustering with batch processing
# clusters = butina_clustering(fingerprint_matrix, threshold=0.7, batch_size=50000)

# # Assign cluster IDs to the molecules in DataFrame
# df['ClusterID'] = -1  # Initialize with -1
# for cluster_id, cluster in enumerate(clusters):
#     for molecule_index in cluster:
#         df.at[molecule_index, 'ClusterID'] = cluster_id
#
# # Save clustered data
# output_path_with_clusters = "./saved_ecfp4_mols_with_clusters_nada.csv"
# df.to_csv(output_path_with_clusters, index=False)

# print(f"Clustered data saved to {output_path_with_clusters}")
# print(df.head())


# DEBUG STATEMENTS:
# df = pd.read_csv(f"./data/trainable_selfies_model_lipinski.csv")
#
# fp = make_fingerprint_thisthat(df)
#
# print(fp)