from generateFingerprints import make_fingerprint_thisthat
from generateClusters import cluster_molecules

import pandas as pd

import yaml

from prepare_dataset import prepare_dataset_for_pretrain

def load_hyperparameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

hyperparameters = load_hyperparameters("./combined_config.yml")
bart_hyperparameters = hyperparameters.get("BART", {})

for key in bart_hyperparameters.keys():
    prepare_dataset_for_pretrain(f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv")

for key in bart_hyperparameters.keys():
    df = pd.read_csv(f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}.csv")
    print(f"reading the csv:\n{df.columns}")
    fp = make_fingerprint_thisthat(df)

    print(fp)
    fp.to_csv(f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}_FP.csv")

for key in bart_hyperparameters.keys():
    df = pd.read_csv(f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}_FP.csv")
    print(f"reading the clustered csv:\n{df.columns}")
    clusters = cluster_molecules(df, f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}_FP_CLUSTERED.csv")
    print(clusters)
    # print(clusters)
    # clusters.to_csv(f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}_FP.csv")
