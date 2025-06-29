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
