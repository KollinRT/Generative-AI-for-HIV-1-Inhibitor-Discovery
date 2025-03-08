from generateClusters import cluster_molecules

import pandas as pd

import yaml

from prepare_dataset import prepare_dataset_for_pretrain

from fp import create_fingerprint

def load_hyperparameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

hyperparameters = load_hyperparameters("./combined_config.yml")
bart_hyperparameters = hyperparameters.get("BART", {})

# Define numPerms and thresholds as lists
# numPerms = [64, 128, 256, 512]
numPerms = [512]

# thresholds_map = {
#     64:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     128: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     256: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     512: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# }
thresholds_map = {
    512: [0.4, 0.5, 0.6]
}


# for key in bart_hyperparameters.keys():
#     prepare_dataset_for_pretrain(f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv")
#
# for key in bart_hyperparameters.keys():
#     df = pd.read_csv(f"./data/trainable_selfies_{key}.csv")
#     print(f"reading the csv:\n{df.columns}")
#     fp = create_fingerprint(df)
#
#     fp = fp.drop(columns=['Molecule'])
#     print(fp)
#     fp.to_csv(f"./data/trainable_selfies_{key}_FP.csv")

# for key in bart_hyperparameters.keys():
#     df = pd.read_csv(f"./data/trainable_selfies_{key}_FP.csv")
#     # print(f"reading the clustered csv:\n{df.columns}")
#     output_filename = f"./data/trainable_selfies_{key}_FP_CLUSTERED_{num_perm}perms_{int(threshold * 10)}.csv"
#
#     clusters = cluster_molecules(df, f"./data/trainable_selfies_{key}_FP_CLUSTERED_512perms_03.csv", num_perm=512, lsh_threshold=0.3)
#     # print(clusters)
#     # print(clusters)
#     # clusters.to_csv(f"/media/kollin/WindowsSecondary1/ThesisBU/Thesis/WIP_Thesis/data/trainable_selfies_{key}_FP.csv")
# Step 3: Perform Clustering using numPerms and thresholds
# for key in bart_hyperparameters.keys():
#     for num_perm in numPerms:
#         for threshold in thresholds_map[num_perm]:
#             df = pd.read_csv(f"./data/trainable_selfies_{key}_FP.csv")
#
#             output_filename = f"./data/trainable_selfies_{key}_FP_CLUSTERED_{num_perm}perms_{int(threshold*10)}.csv"
#             clusters = cluster_molecules(df, output_filename, num_perm=num_perm, lsh_threshold=threshold)
#
#             print(f"Saved clustered file: {output_filename}")

for key in bart_hyperparameters.keys():
    prepare_dataset_for_pretrain(f"./model_name_{key}.csv", f"./data/trainable_selfies_{key}.csv")

for key in bart_hyperparameters.keys():
    df = pd.read_csv(f"./data/trainable_selfies_{key}.csv")
    print(f"reading the csv:\n{df.columns}")
    fp = create_fingerprint(df)

    fp = fp.drop(columns=['Molecule'])
    print(fp)
    fp.to_csv(f"./data/trainable_selfies_{key}_FP.csv")


# Step 3: Perform Clustering & Log Cluster Count
for key in bart_hyperparameters.keys():
    log_filename = f"./data/clustering_summary_{key}.txt"

    # Ensure log file is empty before writing
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Clustering Summary for Config: {key}\n")
        log_file.write("NumPerm\tThreshold\tNumClusters\n")

    for num_perm in numPerms:
        for threshold in thresholds_map[num_perm]:
            df = pd.read_csv(f"./data/trainable_selfies_{key}_FP.csv")
            print(f"Reading the clustered CSV:\n{df.columns}")

            output_filename = f"./data/trainable_selfies_{key}_FP_CLUSTERED_{num_perm}perms_{int(threshold * 10)}.csv"
            clusters = cluster_molecules(df, output_filename, num_perm=num_perm, lsh_threshold=threshold)

            # Count number of clusters
            num_clusters = clusters['Cluster'].nunique() if 'Cluster' in clusters.columns else len(clusters)

            # Log to file
            with open(log_filename, 'a') as log_file:
                log_file.write(f"{num_perm}\t{threshold}\t{num_clusters}\n")

            print(f"Saved clustered file: {output_filename} with {num_clusters} clusters.")

    print(f"Clustering summary saved to: {log_filename}")

# TODO: Just do all at 256 and 0.7..... NEW 03/07/2025 is the one that I utilized!
