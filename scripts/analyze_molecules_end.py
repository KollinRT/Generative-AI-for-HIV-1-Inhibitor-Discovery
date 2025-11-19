import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import Draw

import math

# Settings
# DATA_PATH = "/home/kollin/Desktop/ExploreThesis/WIP_Thesis/benchmark_runs/end_stage_more_tokens_needed/10000_mol_10_batch/generated_10000_molecules_FILTERED.csv"
DATA_PATH = "/home/kollin/Desktop/TestExploreThesis/WIP_Thesis/benchmark_runs/50000_mol_10_batch/generated_50000_molecules_SMILES.csv"

SMILES_COL_CANDIDATES = ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES"]
N_BITS = 2048
RADIUS = 2
TOP_N = 50  # how many top molecules to keep for output/visual
IMG_TOP = 9  # molecules shown in a grid image

# Approved INSTI reference SMILES (isomeric where available)
REFS = {
    "dolutegravir": "C[C@@H]1CCO[C@@H]2N1C(=O)c3c(c(=O)c(cn3C2)C(=O)NCc4ccc(cc4F)F)O",
    "bictegravir": "c1c(cc(c(c1F)CNC(=O)c2cn3c(c(c2=O)O)C(=O)N4[C@H]5CC[C@H](C5)O[C@@H]4C3)F)F",
    "cabotegravir": "C[C@H]1CO[C@H]2N1C(=O)c3c(c(=O)c(cn3C2)C(=O)NCc4ccc(cc4F)F)O",
    "raltegravir": r"Cc1nnc(o1)C(=O)NC(C)(C)C\3=N\C(C(=O)NCc2ccc(F)cc2)=C(\O)C(=O)N/3C",
    "elvitegravir": r"Clc1cccc(c1F)Cc3c(OC)cc2c(C(=O)\C(=C/N2[C@H](CO)C(C)C)C(=O)O)c3",
}

# --- Load data
df = pd.read_csv(DATA_PATH)
# find smiles column
smiles_col = None
for c in df.columns:
    if c in SMILES_COL_CANDIDATES:
        smiles_col = c
        break
if smiles_col is None:
    # fallback: guess by name contains 'smiles'
    for c in df.columns:
        if "smile" in c.lower():
            smiles_col = c
            break

if smiles_col is None:
    raise ValueError(f"Couldn't find a SMILES column. Columns: {df.columns.tolist()}")

df = df.rename(columns={smiles_col: "SMILES"}).copy()
df = df.dropna(subset=["SMILES"])
df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)


# --- Parse molecules
def parse_mol(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


mols = [parse_mol(s) for s in df["SMILES"].tolist()]
valid_mask = [m is not None for m in mols]
df = df.loc[valid_mask].reset_index(drop=True)
mols = [m for m in mols if m is not None]


# --- Compute descriptors
def calc_descriptors(m):
    return {
        "MolWt": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "HBD": Descriptors.NumHDonors(m),
        "HBA": Descriptors.NumHAcceptors(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "RotB": Descriptors.NumRotatableBonds(m),
    }


desc_list = [calc_descriptors(m) for m in mols]
desc_df = pd.DataFrame(desc_list)

# --- Morgan fingerprints for dataset
fps = [GetMorganFingerprintAsBitVect(m, radius=RADIUS, nBits=N_BITS) for m in mols]

# --- Prepare reference molecules, descriptors, fps
ref_mols = {name: Chem.MolFromSmiles(smi) for name, smi in REFS.items()}
# Some references may fail to parse depending on stereochemistry; filter any None
ref_mols = {name: m for name, m in ref_mols.items() if m is not None}
ref_desc = {name: calc_descriptors(m) for name, m in ref_mols.items()}
ref_fps = {
    name: GetMorganFingerprintAsBitVect(m, radius=RADIUS, nBits=N_BITS)
    for name, m in ref_mols.items()
}

ref_desc_df = pd.DataFrame(ref_desc).T  # rows are refs

# --- Compute property-likeness score (Gaussian around ref means)
props = ["MolWt", "LogP", "HBD", "HBA", "TPSA", "RotB"]
mu = ref_desc_df[props].mean()
sigma = ref_desc_df[props].std(ddof=0).replace(0, 1e-6)  # avoid divide by zero


def prop_score(row):
    z2 = (((row[props] - mu) / sigma) ** 2).sum()
    # convert to [0,1] with gaussian-like decay; 6 props -> typical z2 ~ 0..
    return float(math.exp(-z2 / (2 * len(props))))


prop_scores = desc_df.apply(prop_score, axis=1)

# --- Similarity to closest approved INSTI
best_ref = []
best_sim = []

# Precollect ref fps in list for fast BulkTanimoto
ref_names = list(ref_fps.keys())
ref_fp_list = [ref_fps[n] for n in ref_names]

for fp in fps:
    sims = BulkTanimotoSimilarity(fp, ref_fp_list)
    i_best = int(np.argmax(sims))
    best_ref.append(ref_names[i_best])
    best_sim.append(float(sims[i_best]))

# --- Final score combine (weights tuned heuristically)
w_sim, w_prop = 0.7, 0.3
final_score = w_sim * np.array(best_sim) + w_prop * np.array(prop_scores)

# --- Assemble results
out = df.copy()
out["best_ref"] = best_ref
out["sim_to_ref"] = np.round(best_sim, 3)
out = pd.concat([out, desc_df], axis=1)
out["prop_score"] = np.round(prop_scores, 3)
out["final_score"] = np.round(final_score, 3)
# out["final_score"] = np.round(final_score, 30)

# Sort and keep top N
out_top = (
    out.sort_values("final_score", ascending=False).head(TOP_N).reset_index(drop=True)
)

# Save outputs
# CSV_PATH = "/home/kollin/Desktop/ExploreThesis/WIP_Thesis/benchmark_runs/end_stage_more_tokens_needed/10000_mol_10_batch/insti_screen_results.csv"
CSV_PATH = "/home/kollin/Desktop/TestExploreThesis/WIP_Thesis/benchmark_runs/50000_mol_10_batch/insti_screen_results.csv"

out.to_csv(CSV_PATH, index=False)

# # Create a grid image for the very top molecules
# top_mols = [mols[i] for i in out.sort_values("final_score", ascending=False).index[:IMG_TOP]]
# top_labels = [
#     f"{i+1}. score={row.final_score:.2f}\nsim={row.sim_to_ref:.2f} vs {row.best_ref}"
#     for i, row in out_top.iloc[:IMG_TOP].iterrows()
# ]
#
# img = Draw.MolsToGridImage(
#     top_mols,
#     molsPerRow=4,
#     subImgSize=(300,300),
#     legends=top_labels,
#     useSVG=False
# )
# IMG_PATH = "/home/kollin/Desktop/ExploreThesis/WIP_Thesis/benchmark_runs/end_stage_more_tokens_needed/10000_mol_10_batch/top_insti_like.png"
# img.save(IMG_PATH)

# Create a grid image for the very top molecules
out_sorted = out.sort_values("final_score", ascending=False).head(IMG_TOP)
top_mols = [mols[i] for i in out_sorted.index]

top_labels = [
    # f"{i+1}. score={row.final_score:.2f} | sim={row.sim_to_ref:.2f} vs {row.best_ref}\n{row.SMILES}"
    f"{i + 1}. score={row.final_score:.2f} | sim={row.sim_to_ref:.2f} vs {row.best_ref}"
    for i, row in out_sorted.iterrows()
]

img = Draw.MolsToGridImage(
    top_mols,
    molsPerRow=3,
    subImgSize=(400, 400),  # bump size to fit SMILES text
    legends=top_labels,
    useSVG=False,
)

# IMG_PATH = "/home/kollin/Desktop/ExploreThesis/WIP_Thesis/benchmark_runs/end_stage_more_tokens_needed/10000_mol_10_batch/top_insti_like.png"

IMG_PATH = "/home/kollin/Desktop/TestExploreThesis/WIP_Thesis/benchmark_runs/50000_mol_10_batch/top_insti_like.png"
img.save(IMG_PATH)


# # Preview to user
# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("Top INSTI-like candidates (preview)", out_top)
#
# CSV_PATH, IMG_PATH
