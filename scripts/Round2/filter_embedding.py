import selfies as sf
import pandas as pd

# Define embedding length here rather than in model
embed_length = 514
max_selfies_length = embed_length - 2  # Account for BOS and EOS tokens

# Load the dataset
df = pd.read_csv("combined_selfies_dataset.csv")
#df = pd.read_csv("combined_selfies_dataset_pretrain.csv")

# Make sure whole column is just string
df["selfies"] = df["selfies"].astype(str)

# Add a column with the SELFIES token lengths
df["selfies_length"] = df["selfies"].apply(lambda x: len(list(sf.split_selfies(x))))

# Sort to find the longest
df_sorted = df.sort_values(by="selfies_length", ascending=False)

# Show the top 5 longest sequences
print(df_sorted[["selfies", "selfies_length"]].head(5).to_string(index=False))


# TODO: Get model max positional embedding then get selfies length from here
# Make sure length is no more than max positional embedding - 2... +2 for bos/eos tokens.
# This should be easy enough to get from config file... Then save into same file *_CLEANED .csv

df_cleaned = df[df["selfies_length"] <= max_selfies_length]

# Keep only the 'selfies' column (as a DataFrame, not Series)
df_cleaned = df_cleaned[["selfies"]]

# Save the cleaned dataset
df_cleaned.to_csv(f"./combined_selfies_dataset_embedding_{str(embed_length)}.csv", index=False)

print(f"Filtered dataset saved. Original size: {len(df)}, Cleaned size: {len(df_cleaned)}")

# combined_selfies_dataset_embedding_514
