# # # import pandas as pd
# # # import networkx as nx
# # # from datasketch import MinHash, MinHashLSH
# # # import time
# # #
# # #
# # # """
# # # TODO: NEW
# # # - ADD THE GENERATION OF FINGERPRINTS INTO THIS SCRIPT... THIS WOULD THEN BE CONTINUED BY THE bitstring_to_set code...
# # # - - The generation of the fingerprints is done somewhere else...
# # # """
# # # # -------------------------------
# # # # Parameters – adjust as needed:
# # # # -------------------------------
# # # num_perm = 256  # Number of permutations for MinHash (more => higher accuracy, but slower)
# # # lsh_threshold = 0.7  # LSH similarity threshold (approximate Tanimoto/Jaccard similarity)
# # #
# # #
# # # # Note: Tanimoto similarity for binary fingerprints is equivalent to the Jaccard similarity.
# # #
# # # # -------------------------------
# # # # Helper Function
# # # # -------------------------------
# # # def bitstring_to_set(bitstring):
# # #     """
# # #     Convert a fingerprint bit string (e.g. '001101...') to a set of indices where the bit is '1'.
# # #     For example, '0101' becomes {1, 3} (0-indexed).
# # #     """
# # #     return {i for i, bit in enumerate(bitstring) if bit == '1'}
# # #
# # #
# # # # -------------------------------
# # # # 1. Load the CSV with Precomputed Fingerprints
# # # # -------------------------------
# # # df = pd.read_csv("molecule_fingerprints_no_address.csv")
# # # n = len(df)
# # # print(f"Total number of molecules: {n}")
# # #
# # # # For demonstration, you might try a smaller subset first:
# # # # df = df.head(10000)
# # # df = df.head(100)
# # # # n = len(df)
# # #
# # # # -------------------------------
# # # # 2. Convert Fingerprints to Sets and Create MinHash Signatures
# # # # -------------------------------
# # # start = time.time()
# # # fingerprint_sets = []
# # # minhashes = []
# # #
# # # # Process each fingerprint.
# # # # (For 1M molecules, you might want to use parallel processing or process in batches.)
# # #     for idx, fp in enumerate(df['Fingerprint']):
# # #     s = bitstring_to_set(fp)
# # #     fingerprint_sets.append(s)
# # #
# # #     m = MinHash(num_perm=num_perm)
# # #     # Update the MinHash with each element in the set.
# # #     for item in s:
# # #         # Convert the item to bytes (here, we use the string of the integer index).
# # #         m.update(str(item).encode('utf8'))
# # #     minhashes.append(m)
# # #     print(minhashes)
# # #
# # #     if (idx + 1) % 10000 == 0:
# # #         print(f"Processed {idx + 1} fingerprints...")
# # #
# # # print(f"Created MinHash signatures for {n} molecules in {time.time() - start:.1f} seconds.")
# # #
# # # # -------------------------------
# # # # 3. Build the LSH Index and Insert the MinHashes
# # # # -------------------------------
# # # lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
# # # for i, m in enumerate(minhashes):
# # #     lsh.insert(f"mol_{i}", m)
# # # print("LSH index built.")
# # # print(lsh)
# # #
# # # # -------------------------------
# # # # 4. Build a Similarity Graph Based on LSH Neighbors
# # # # -------------------------------
# # # # Create an undirected graph where each node is a molecule (index),
# # # # and an edge indicates that two molecules are approximate neighbors.
# # # G = nx.Graph()
# # # G.add_nodes_from(range(n))
# # #
# # # start = time.time()
# # # for i, m in enumerate(minhashes):
# # #     # Query LSH for neighbors; keys are strings like "mol_{j}"
# # #     neighbors = lsh.query(m)
# # #     # Add an edge for each neighbor (avoid self-loops and duplicate edges)
# # #     for nb in neighbors:
# # #         j = int(nb.split('_')[1])
# # #         if i < j:
# # #             G.add_edge(i, j)
# # #     if (i + 1) % 10000 == 0:
# # #         print(f"Graph: processed {i + 1} molecules...")
# # # print(f"Graph built in {time.time() - start:.1f} seconds.")
# # #
# # # # -------------------------------
# # # # 5. Extract Clusters from the Graph
# # # # -------------------------------
# # # # Each connected component in the graph is treated as a cluster.
# # # clusters = list(nx.connected_components(G))
# # # print(f"Number of clusters found: {len(clusters)}")
# # #
# # # # -------------------------------
# # # # 6. Create a Cluster Mapping and Merge with the Original DataFrame
# # # # -------------------------------
# # # cluster_mapping = {"OriginalIndex": [], "Cluster": []}
# # # for cluster_id, cluster in enumerate(clusters):
# # #     for idx in cluster:
# # #         cluster_mapping["OriginalIndex"].append(idx)
# # #         cluster_mapping["Cluster"].append(cluster_id)
# # #
# # # cluster_df = pd.DataFrame(cluster_mapping)
# # # cluster_df.sort_values("OriginalIndex", inplace=True)
# # #
# # # # Merge with the original DataFrame (assuming the DataFrame index is the molecule ID).
# # # df_with_clusters = df.merge(cluster_df, left_index=True, right_on="OriginalIndex", how="left")
# # #
# # # # Save the merged DataFrame to a CSV file.
# # # # output_csv = f"molecule_with_clusters_lsh_{str(num_perm)}perm.csv"
# # # output_csv = "./data/trainable_selfies_lipinski.csv"
# # #
# # # df_with_clusters.to_csv(output_csv, index=False)
# # # print(f"Cluster assignments saved to {output_csv}")
# #
# # import pandas as pd
# # import networkx as nx
# # from datasketch import MinHash, MinHashLSH
# # import time
# #
# #
# # def cluster_molecules(
# #         df,
# #         output_csv,
# #         num_perm=256,
# #         lsh_threshold=0.7,
# # ):
# #     """
# #     Clusters molecules based on precomputed fingerprint strings using MinHashLSH.
# #
# #     Parameters:
# #     -----------
# #     # input_csv : str
# #     #     Path to the CSV file that contains a column named 'Fingerprint'.
# #     output_csv : str
# #         Path to save the output CSV with cluster assignments.
# #     num_perm : int, optional (default=256)
# #         Number of permutations for MinHash (more => higher accuracy, but slower).
# #     lsh_threshold : float, optional (default=0.7)
# #         LSH similarity threshold (approximate Tanimoto/Jaccard similarity).
# #     # subset_size : int or None, optional (default=None)
# #     #     If provided, only process the first `subset_size` molecules.
# #
# #     Returns:
# #     --------
# #     df_with_clusters : pandas.DataFrame
# #         The original DataFrame merged with cluster assignments.
# #     clusters : list of sets
# #         List of clusters (each cluster is a set of row indices).
# #     """
# #
# #     def bitstring_to_set(bitstring):
# #         """
# #         Convert a fingerprint bit string (e.g. '001101...') to a set of indices where the bit is '1'.
# #         For example, '0101' becomes {1, 3} (0-indexed).
# #         """
# #         return {i for i, bit in enumerate(bitstring) if bit == '1'}
# #
# #     # -------------------------------
# #     # 1. Load the CSV with Precomputed Fingerprints
# #     # -------------------------------
# #     # df = pd.read_csv(input_csv)
# #     n = len(df)
# #     print(f"Total number of molecules: {n}")
# #
# #     # -------------------------------
# #     # 2. Convert Fingerprints to Sets and Create MinHash Signatures
# #     # -------------------------------
# #     start = time.time()
# #     fingerprint_sets = []
# #     minhashes = []
# #
# #     for idx, fp in enumerate(df['Fingerprint']):
# #         s = bitstring_to_set(fp)
# #         fingerprint_sets.append(s)
# #
# #         m = MinHash(num_perm=num_perm)
# #         # Update the MinHash with each element in the set.
# #         for item in s:
# #             m.update(str(item).encode('utf8'))
# #         minhashes.append(m)
# #
# #         if (idx + 1) % 10000 == 0:
# #             print(f"Processed {idx + 1} fingerprints...")
# #
# #     elapsed = time.time() - start
# #     print(f"Created MinHash signatures for {n} molecules in {elapsed:.1f} seconds.")
# #
# #     # -------------------------------
# #     # 3. Build the LSH Index and Insert the MinHashes
# #     # -------------------------------
# #     lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
# #     for i, m in enumerate(minhashes):
# #         lsh.insert(f"mol_{i}", m)
# #     print("LSH index built.")
# #
# #     # -------------------------------
# #     # 4. Build a Similarity Graph Based on LSH Neighbors
# #     # -------------------------------
# #     G = nx.Graph()
# #     G.add_nodes_from(range(n))
# #
# #     start = time.time()
# #     for i, m in enumerate(minhashes):
# #         # Query LSH for neighbors; keys are strings like "mol_{j}"
# #         neighbors = lsh.query(m)
# #         # Add an edge for each neighbor (avoid self-loops and duplicate edges)
# #         for nb in neighbors:
# #             j = int(nb.split('_')[1])
# #             if i < j:
# #                 G.add_edge(i, j)
# #
# #         if (i + 1) % 10000 == 0:
# #             print(f"Graph: processed {i + 1} molecules...")
# #
# #     elapsed = time.time() - start
# #     print(f"Graph built in {elapsed:.1f} seconds.")
# #
# #     # -------------------------------
# #     # 5. Extract Clusters from the Graph
# #     # -------------------------------
# #     clusters = list(nx.connected_components(G))
# #     print(f"Number of clusters found: {len(clusters)}")
# #
# #     # -------------------------------
# #     # 6. Create a Cluster Mapping and Merge with the Original DataFrame
# #     # -------------------------------
# #     cluster_mapping = {"OriginalIndex": [], "Cluster": []}
# #     for cluster_id, cluster in enumerate(clusters):
# #         for idx in cluster:
# #             cluster_mapping["OriginalIndex"].append(idx)
# #             cluster_mapping["Cluster"].append(cluster_id)
# #
# #     cluster_df = pd.DataFrame(cluster_mapping)
# #     cluster_df.sort_values("OriginalIndex", inplace=True)
# #
# #     # Remove previously merged cluster columns if they exist.
# #     cols_to_drop = [col for col in df.columns if 'OriginalIndex' in col or 'Cluster' in col]
# #     df = df.drop(columns=cols_to_drop, errors='ignore')
# #
# #     # Merge with the original DataFrame (assuming the DataFrame index is the molecule ID).
# #     df_with_clusters = df.merge(cluster_df, left_index=True, right_on="OriginalIndex", how="left")
# #
# #     # Save the merged DataFrame to a CSV file.
# #     df_with_clusters.to_csv(output_csv, index=False)
# #     print(f"Cluster assignments saved to {output_csv}")
# #
# #     return df_with_clusters
# import pandas as pd
# import networkx as nx
# from datasketch import MinHash, MinHashLSH
# import time
#
# def cluster_molecules(df, output_csv, num_perm=256, lsh_threshold=0.7):
#     """
#     Clusters molecules based on precomputed fingerprint bit strings using MinHashLSH.
#
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame that must contain a 'Fingerprint' column with fingerprint bit strings.
#     output_csv : str
#         Path to save the output CSV with cluster assignments.
#     num_perm : int, optional (default=256)
#         Number of permutations for MinHash (more permutations → higher accuracy but slower).
#     lsh_threshold : float, optional (default=0.7)
#         LSH similarity threshold (approximate Tanimoto/Jaccard similarity).
#
#     Returns:
#     --------
#     df : pandas.DataFrame
#         The original DataFrame with an added 'Cluster' column.
#     clusters : list of sets
#         A list of clusters, where each cluster is a set of row indices.
#     """
#
#     def bitstring_to_set(bitstring):
#         """Convert a fingerprint bit string (e.g. '001101...') into a set of indices where the bit is '1'."""
#         return {i for i, bit in enumerate(bitstring) if bit == '1'}
#
#     n = len(df)
#     print(f"Total number of molecules: {n}")
#
#     # 1. Create MinHash signatures from the fingerprint bit strings.
#     start = time.time()
#     minhashes = []
#     for idx, fp in enumerate(df['Fingerprint']):
#         s = bitstring_to_set(fp)
#         m = MinHash(num_perm=num_perm)
#         for item in s:
#             m.update(str(item).encode('utf8'))
#         minhashes.append(m)
#         if (idx + 1) % 10000 == 0:
#             print(f"Processed {idx + 1} fingerprints...")
#     elapsed = time.time() - start
#     print(f"Created MinHash signatures for {n} molecules in {elapsed:.1f} seconds.")
#
#     # 2. Build the LSH index.
#     lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
#     for i, m in enumerate(minhashes):
#         lsh.insert(f"mol_{i}", m)
#     print("LSH index built.")
#
#     # 3. Build a similarity graph using the LSH neighbors.
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     start = time.time()
#     for i, m in enumerate(minhashes):
#         neighbors = lsh.query(m)
#         for nb in neighbors:
#             j = int(nb.split('_')[1])
#             if i < j:  # Avoid self-loops and duplicate edges.
#                 G.add_edge(i, j)
#     elapsed = time.time() - start
#     print(f"Graph built in {elapsed:.1f} seconds.")
#
#     # 4. Extract clusters from the graph (each connected component is a cluster).
#     clusters = list(nx.connected_components(G))
#     print(f"Number of clusters found: {len(clusters)}")
#
#     # 5. Build a mapping from the original DataFrame index to a cluster label.
#     # If a row's index is not in any cluster, assign -1.
#     cluster_mapping = {}
#     for cluster_id, cluster in enumerate(clusters):
#         for idx in cluster:
#             cluster_mapping[idx] = cluster_id
#
#     # Directly assign the cluster labels to a new 'Cluster' column.
#     df['Cluster'] = df.index.map(lambda idx: cluster_mapping.get(idx, -1))
#
#     # 6. Save the resulting DataFrame.
#     df.to_csv(output_csv, index=False)
#     print(f"Cluster assignments saved to {output_csv}")
#
#     return df_with_clusters
import pandas as pd
import networkx as nx
from datasketch import MinHash, MinHashLSH
import time

def cluster_molecules(df, output_csv, num_perm=256, lsh_threshold=0.7):
    """
    Clusters molecules based on precomputed fingerprint bit strings using MinHashLSH.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame that must contain a 'Fingerprint' column with fingerprint bit strings.
    output_csv : str
        Path to save the output CSV with cluster assignments.
    num_perm : int, optional (default=256)
        Number of permutations for MinHash (more permutations → higher accuracy but slower).
    lsh_threshold : float, optional (default=0.7)
        LSH similarity threshold (approximate Tanimoto/Jaccard similarity).

    Returns:
    --------
    df_with_clusters : pandas.DataFrame
        The original DataFrame with an added 'Cluster' column.
    """

    def bitstring_to_set(bitstring):
        """Convert a fingerprint bit string (e.g. '001101...') into a set of indices where the bit is '1'."""
        return {i for i, bit in enumerate(bitstring) if bit == '1'}

    n = len(df)
    print(f"Total number of molecules: {n}")

    # 1. Create MinHash signatures from the fingerprint bit strings.
    start = time.time()
    minhashes = []
    for idx, fp in enumerate(df['Fingerprint']):
        s = bitstring_to_set(fp)
        m = MinHash(num_perm=num_perm)
        for item in s:
            m.update(str(item).encode('utf8'))
        minhashes.append(m)
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} fingerprints...")
    elapsed = time.time() - start
    print(f"Created MinHash signatures for {n} molecules in {elapsed:.1f} seconds.")

    # 2. Build the LSH index.
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(f"mol_{i}", m)
    print("LSH index built.")

    # 3. Build a similarity graph using the LSH neighbors.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    start = time.time()
    for i, m in enumerate(minhashes):
        neighbors = lsh.query(m)
        for nb in neighbors:
            j = int(nb.split('_')[1])
            if i < j:  # Avoid self-loops and duplicate edges.
                G.add_edge(i, j)
    elapsed = time.time() - start
    print(f"Graph built in {elapsed:.1f} seconds.")

    # 4. Extract clusters from the graph (each connected component is a cluster).
    clusters = list(nx.connected_components(G))
    print(f"Number of clusters found: {len(clusters)}")

    # 5. Build a mapping from the original DataFrame index to a cluster label.
    # If a row's index is not in any cluster, assign -1.
    cluster_mapping = {}
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_mapping[idx] = cluster_id

    # Directly assign the cluster labels to a new 'Cluster' column.
    df['Cluster'] = df.index.map(lambda idx: cluster_mapping.get(idx, -1))

    # 6. Save the resulting DataFrame.
    df.to_csv(f"{output_csv[:-4]}_clustered.csv", index=False)
    print(f"Cluster assignments saved to {output_csv[:-4]}_clustered.csv")

    return df
