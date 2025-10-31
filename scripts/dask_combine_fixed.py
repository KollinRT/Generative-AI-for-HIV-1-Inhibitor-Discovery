import dask.dataframe as dd
import selfies as sf  # Assuming your selfies package is called `selfies`

# Load all CSVs
df = dd.read_csv('./data/*.csv')

# Drop unnecessary columns
df = df.drop(['smiles', 'zinc_id'], axis=1)

# Define the token counting function
def compute_token_count(selfies_str):
    try:
        tokens = list(sf.split_selfies(str(selfies_str)))
        return len(tokens)
    except Exception:
        return 0

# Apply compute_token_count on the 'selfies' column
df['token_count'] = df['selfies'].map_partitions(
    lambda part: part.map(compute_token_count),
    meta=('token_count', 'int')
)

# Save as Parquet
df.to_parquet('processed_data_folder_fixed/', overwrite=True)

