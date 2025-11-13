# file: convert_with_pandarallel.py
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
import selfies as sf
import os

# init - choose number of workers; progress_bar optional
pandarallel.initialize(
    nb_workers=32, progress_bar=True
)  # tune nb_workers (e.g. 8 or os.cpu_count()-1)


def process_row(row):
    smiles = row["smiles"]
    zinc_id = row["zinc_id"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        selfies_str = sf.encoder(smiles)
        return pd.Series({"smiles": smiles, "zinc_id": zinc_id, "selfies": selfies_str})
    except Exception:
        return None


def convert_file(infile, out_csv, chunksize=1_000_000):
    # reader assumes whitespace-delimited two columns: SMILES and ZINC ID
    reader = pd.read_csv(
        infile,
        sep=r"\s+",
        names=["smiles", "zinc_id"],
        header=None,
        chunksize=chunksize,
        iterator=True,
    )
    first_write = True
    for i, chunk in enumerate(reader):
        # parallel apply across rows
        results = chunk.parallel_apply(process_row, axis=1)
        # drop None rows
        results = results.dropna(subset=["selfies"])
        # write (append after first)
        if first_write:
            results.to_csv(out_csv, index=False, mode="w")
            first_write = False
        else:
            results.to_csv(out_csv, index=False, header=False, mode="a")
        print(f"Chunk {i} done, wrote {len(results)} records")


if __name__ == "__main__":
    convert_file("zinc15_sampled_10M.smi", "zinc15_10M_selfies_par.csv")
