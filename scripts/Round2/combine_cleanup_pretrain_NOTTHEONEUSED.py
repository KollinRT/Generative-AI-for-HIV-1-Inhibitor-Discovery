import pandas as pd

# Load both CSV files
df1 = pd.read_csv("model_base_selfies_only.csv")
df2 = pd.read_csv("selfies_only_10M_zinc.csv")

# Append (concatenate)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new file (or overwrite the first)
combined_df.to_csv("combined_selfies_dataset_pretrain.csv", index=False)

print("✅ Files successfully combined and saved as combined_selfies_dataset_pretrain.csv")
