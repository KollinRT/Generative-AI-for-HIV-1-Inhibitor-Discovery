from generateClusters import cluster_molecules

import pandas as pd

import yaml


# from fp import create_fingerprint
from generateFingerprints import make_fingerprint_thisthat

import gc  # clear memory


def load_hyperparameters(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


hyperparameters = load_hyperparameters("./combined_config.yml")
bart_hyperparameters = hyperparameters.get("BART", {})

# Define numPerms and thresholds as lists
numPerms = [256]

thresholds_map = {256: [0.7]}

gc.collect()


for key in bart_hyperparameters.keys():
    # df = pd.read_csv(f"./data/trainable_selfies_{key}.csv")
    df = pd.read_csv("combined_selfies_dataset_embedding_514.csv")
    print(f"reading the csv:\n{df.columns}")
    fp = make_fingerprint_thisthat(df)

    fp = fp.drop(columns=["Molecule"])
    print(fp)
    fp.to_csv(f"./data/trainable_selfies_{key}_FP.csv")

gc.collect()

# Step 3: Perform Clustering & Log Cluster Count
for key in bart_hyperparameters.keys():
    log_filename = f"./data/clustering_summary_{key}.txt"

    # Ensure log file is empty before writing
    with open(log_filename, "w") as log_file:
        log_file.write(f"Clustering Summary for Config: {key}\n")
        log_file.write("NumPerm\tThreshold\tNumClusters\n")

    for num_perm in numPerms:
        for threshold in thresholds_map[num_perm]:
            df = pd.read_csv(f"./data/trainable_selfies_{key}_FP.csv")
            print(f"Reading the clustered CSV:\n{df.columns}")

            output_filename = f"./data/trainable_selfies_{key}_FP_CLUSTERED_{num_perm}perms_{int(threshold * 10)}.csv"
            clusters = cluster_molecules(
                df, output_filename, num_perm=num_perm, lsh_threshold=threshold
            )

            # Count number of clusters
            num_clusters = (
                clusters["Cluster"].nunique()
                if "Cluster" in clusters.columns
                else len(clusters)
            )

            # Log to file
            with open(log_filename, "a") as log_file:
                log_file.write(f"{num_perm}\t{threshold}\t{num_clusters}\n")

            print(
                f"Saved clustered file: {output_filename} with {num_clusters} clusters."
            )

    print(f"Clustering summary saved to: {log_filename}")
