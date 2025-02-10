# TODO NEW Winter Quarter: Talk to Fabry about scaffolds? I was thinking scaffolds and bias temperature to explore new novel
# structure space... But how do we do this? Lets find another way to pattern mine similarities for training or look more
# into Murcko scaffolds! I like the translate SELFIES to SMILES for this and get the list... but now I need to run and see
# How many scaffold themes that I get!
#
# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
#
# def extract_scaffold(smiles):
#     """Extract the Bemis-Murcko scaffold from a SMILES string."""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#     return Chem.MolToSmiles(scaffold)
#
#
# def calculate_scaffold_diversity(smiles_list):
#     """Calculate the scaffold diversity for a list of SMILES."""
#     scaffolds = [extract_scaffold(smiles) for smiles in smiles_list]
#     unique_scaffolds = set(scaffolds)
#     return len(unique_scaffolds) / len(scaffolds)
#
#
# import matplotlib.pyplot as plt
# from collections import Counter
#
#
# def plot_scaffold_distribution(smiles_list):
#     scaffolds = [extract_scaffold(smiles) for smiles in smiles_list]
#     scaffold_counts = Counter(scaffolds)
#     most_common_scaffolds = scaffold_counts.most_common(20)
#
#     labels, values = zip(*most_common_scaffolds)
#     plt.figure(figsize=(10, 5))
#     plt.bar(range(len(labels)), values, tick_label=labels)
#     plt.xticks(rotation=90)
#     plt.xlabel("Scaffold")
#     plt.ylabel("Frequency")
#     plt.title("Top 20 Most Common Scaffolds")
#     plt.show()
#
# def jaccard_index(set_a, set_b):
#     """Calculate the Jaccard index between two sets."""
#     intersection = len(set_a.intersection(set_b))
#     union = len(set_a.union(set_b))
#     return intersection / union
#
#
# # from rdkit import Chem
# # from rdkit.Chem.Scaffolds import MurckoScaffold
# # import random
# #
# # def scaffold_split(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1):
# #     """Split dataset into training, validation, and test sets based on scaffolds."""
# #     scaffolds = {}
# #     for smiles in dataset:
# #         mol = Chem.MolFromSmiles(smiles)
# #         if mol:
# #             scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
# #             if scaffold not in scaffolds:
# #                 scaffolds[scaffold] = []
# #             scaffolds[scaffold].append(smiles)
# #
# #     # Shuffle and split scaffolds
# #     all_scaffolds = list(scaffolds.keys())
# #     random.shuffle(all_scaffolds)
# #     n_total = len(all_scaffolds)
# #     train_cutoff = int(n_total * train_frac)
# #     val_cutoff = int(n_total * (train_frac + val_frac))
# #
# #     train_scaffolds = all_scaffolds[:train_cutoff]
# #     val_scaffolds = all_scaffolds[train_cutoff:val_cutoff]
# #     test_scaffolds = all_scaffolds[val_cutoff:]
# #
# #     # Assign molecules based on scaffold splits
# #     train_set = [m for s in train_scaffolds for m in scaffolds[s]]
# #     val_set = [m for s in val_scaffolds for m in scaffolds[s]]
# #     test_set = [m for s in test_scaffolds for m in scaffolds[s]]
# #
# #     return train_set, val_set, test_set
#
# # Random Splitting (When Scaffold Splitting is Not Feasible)
# # For finetuning
# # Use the test set strictly for final evaluation.
#
#
from rdkit import Chem
import selfies as sf

def selfies_to_mol(selfies_string):
    """Convert a SELFIES string to an RDKit Mol object."""
    smiles = sf.decoder(selfies_string)  # Decode SELFIES to SMILES
    return Chem.MolFromSmiles(smiles)    # Convert SMILES to RDKit molecule
#
#
from rdkit.Chem.Scaffolds import MurckoScaffold

def extract_scaffold_from_selfies(selfies_string):
    """Extract the Bemis-Murcko scaffold from a SELFIES string."""
    mol = selfies_to_mol(selfies_string)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)
#
#
# import random
#
#
# def scaffold_split_selfies(selfies_list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
#     """Split a dataset of SELFIES strings into training, validation, and test sets based on scaffolds."""
#     scaffolds = {}
#     for selfies in selfies_list:
#         scaffold = extract_scaffold_from_selfies(selfies)
#         if scaffold:
#             if scaffold not in scaffolds:
#                 scaffolds[scaffold] = []
#             scaffolds[scaffold].append(selfies)
#
#     # Shuffle and split scaffolds
#     all_scaffolds = list(scaffolds.keys())
#     random.shuffle(all_scaffolds)
#     n_total = len(all_scaffolds)
#     train_cutoff = int(n_total * train_frac)
#     val_cutoff = int(n_total * (train_frac + val_frac))
#
#     train_scaffolds = all_scaffolds[:train_cutoff]
#     val_scaffolds = all_scaffolds[train_cutoff:val_cutoff]
#     test_scaffolds = all_scaffolds[val_cutoff:]
#
#     # Assign SELFIES strings to splits
#     train_set = [s for sc in train_scaffolds for s in scaffolds[sc]]
#     val_set = [s for sc in val_scaffolds for s in scaffolds[sc]]
#     test_set = [s for sc in test_scaffolds for s in scaffolds[sc]]
#
#     return train_set, val_set, test_set
#
#
# def calculate_scaffold_overlap(set_a, set_b):
#     scaffolds_a = {extract_scaffold_from_selfies(s) for s in set_a}
#     scaffolds_b = {extract_scaffold_from_selfies(s) for s in set_b}
#     intersection = len(scaffolds_a.intersection(scaffolds_b))
#     union = len(scaffolds_a.union(scaffolds_b))
#     return intersection / union
# # overlap = calculate_scaffold_overlap(train_set, val_set)
# # print(f"Train-Validation Scaffold Overlap: {overlap:.2f}")
#
#
# # selfies_data = [
# #     "CCO",
# #     "CCN",
# #     "CCCO",
# #     "CCCN"
# # ]
# #
# # # Scaffold splitting
# # train_set, val_set, test_set = scaffold_split_selfies(selfies_data)
# #
# # print("Training Set:", train_set)
# # print("Validation Set:", val_set)
# # print("Test Set:", test_set)
#

drugs = [
    "[C][C][N][C][=C][C][Branch1][C][C][=N][C][Branch2][Ring1][Branch2][N][C][C][N][C][=Branch1][C][=O][N][C][=C][C][Branch1][C][F][=C][C][Branch1][C][F][=C][Ring1][Branch2][=N][Ring2][Ring1][=Branch1]",
    "[C][C][C][C][=Branch1][C][=O][N][C][C][O][C][C][Ring1][Branch1][C][C][=C][C][=N][C][=C][Ring1][=Branch1]",
    "[O][=C][Branch2][Ring1][Ring2][N][N][=N][C][=C][C][Branch1][C][Cl][=C][Branch1][C][Cl][C][=C][Ring1][Branch2][Ring1][O][C][Branch1][C][F][Branch1][C][F][F]",
    "[C][C][=C][N][=C][C][=C][Ring1][=Branch1][N][C][C][N][C][C][=C][C][=C][Branch1][#Branch1][O][C][Branch1][C][C][C][C][Branch1][C][Cl][=C][Ring1][O]"
]
for drug in drugs:
    print(extract_scaffold_from_selfies(drug))

# def editDistRec(s1, s2, m, n):
#     # If first string is empty, the only option is to
#     # insert all characters of second string into first
#     if m == 0:
#         return n
#
#     # If second string is empty, the only option is to
#     # remove all characters of first string
#     if n == 0:
#         return m
#
#     # If last characters of two strings are same, nothing
#     # much to do. Get the count for
#     # remaining strings.
#     if s1[m - 1] == s2[n - 1]:
#         return editDistRec(s1, s2, m - 1, n - 1)
#
#     # If last characters are not same, consider all three
#     # operations on last character of first string,
#     # recursively compute minimum cost for all three
#     # operations and take minimum of three values.
#     return 1 + min(editDistRec(s1, s2, m, n - 1),    # Insert
#                    editDistRec(s1, s2, m - 1, n),    # Remove
#                    editDistRec(s1, s2, m - 1, n - 1) # Replace
#                    )
#
# def editDist(s1, s2):
#     return editDistRec(s1, s2, len(s1), len(s2))

def edit_dist(s1, s2):
    m, n = len(s1), len(s2)
    prev = 0  # Stores dp[i-1][j-1]
    curr = list(range(n + 1))  # Stores dp[i][j-1] and dp[i][j]

    for i in range(1, m + 1):
        prev = curr[0]
        curr[0] = i
        for j in range(1, n + 1):
            temp = curr[j]
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev
            else:
                curr[j] = 1 + min(curr[j - 1], prev, curr[j])
            prev = temp
    return curr[n]


string1 = "O=C(NCCNc1ncccn1)Nc1ccccc1"
string2 = "c1cc(CC2CCOC2)ccn1"

print(edit_dist(string1, string2))

#TODO: 01/01/2025:
"""
- Get some scaffolds and then calculate the word difference values?
"""

# TODO: 01/08/2025
"""
- Think of how to do scaffold levels...
- granularity... think of the paper!
- Try to do a SOM cluster of the 940k molecules? See the space?
- - Quantify edit distance as the metric for similarity?
"""