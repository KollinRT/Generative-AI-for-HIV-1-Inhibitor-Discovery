import pandas as pd
from tqdm import tqdm
import selfies as sf
from rdkit import Chem


# Read SMILES and ZINC ID
def read_smiles_file(filename):
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                yield parts[0], parts[1]


# Convert to SELFIES
def process_smiles(smiles, zinc_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        selfies_str = sf.encoder(smiles)
        return {"smiles": smiles, "zinc_id": zinc_id, "selfies": selfies_str}
    except Exception:
        return None


# Process and store
data = []
for smiles, zinc_id in tqdm(read_smiles_file("zinc15_all_raw.smi"), total=10_000_000):
    record = process_smiles(smiles, zinc_id)
    if record:
        data.append(record)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("zinc15_all_selfies.csv", index=False)
