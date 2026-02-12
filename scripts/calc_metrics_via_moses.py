# import pandas as pd
# from moses.metrics import get_all_metrics
#
# # Load your SMILES
# df = pd.read_csv("generated_10000_molecules_smiles.csv")
# gen_smiles = df["smiles"].dropna().astype(str).tolist()

import pandas as pd
from moses.metrics import get_all_metrics

# -----------------------
# 1. Load your generated molecules
# -----------------------
gen = (
    pd.read_csv(
        "/home/kollin/Desktop/TestExploreThesis/WIP_Thesis/benchmark_runs/10000_mol_10_batch/generated_10000_molecules_smiles.csv"
    )["smiles"]
    .dropna()
    .astype(str)
    .tolist()
)

# -----------------------
# 2. Load MOSES reference splits
# -----------------------
ref_test = (
    pd.read_csv("/home/kollin/Desktop/GithubProj/kaggle/moses/test.csv")["SMILES"]
    .dropna()
    .astype(str)
    .tolist()
)
ref_train = (
    pd.read_csv("/home/kollin/Desktop/GithubProj/kaggle/moses/train.csv")["SMILES"]
    .dropna()
    .astype(str)
    .tolist()
)

# -----------------------
# 3. Compute MOSES metrics
# -----------------------
metrics = get_all_metrics(gen=gen, test=ref_test, k=ref_train, n_jobs=8)

# -----------------------
# 4. Save results
# -----------------------
import json

with open("moses_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame([metrics]).to_csv("moses_metrics.csv", index=False)

print("Saved moses_metrics.json and moses_metrics.csv")
