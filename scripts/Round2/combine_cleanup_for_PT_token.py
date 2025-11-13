import pandas as pd

df1 = pd.read_csv("../../model_name_model_4_warmup_selfies.csv")
df1 = df1[["selfies"]]
df1.to_csv("model_base_selfies_only.csv", index=False)

df2 = pd.read_csv("zinc15_10M_selfies_par.csv")
df2 = df2[["selfies"]]
df2.to_csv("selfies_only_10M_zinc.csv", index=False)

df3 = pd.read_csv("smiles_finetune_data_properties_selfies.csv")
df3 = df3[["selfies"]]
df3.to_csv("selfies_only_finetune_data.csv", index=False)

# Load both CSV files
df1 = pd.read_csv("model_base_selfies_only.csv")
df2 = pd.read_csv("selfies_only_10M_zinc.csv")
df3 = pd.read_csv("selfies_only_finetune_data.csv")

# Append (concatenate)
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save to a new file (or overwrite the first)
combined_df.to_csv("combined_selfies_dataset.csv", index=False)

print("Files successfully combined and saved as combined_selfies_dataset.csv")