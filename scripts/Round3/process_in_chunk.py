import csv
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Read generator
def read_smiles_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                yield parts[0], parts[1]

# Conversion function
def process_smiles(record):
    smiles, zinc_id = record
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        selfies_str = sf.encoder(smiles)
        return [smiles, zinc_id, selfies_str]
    except Exception:
        return None

# Batch generator
def batch_iterator(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Process all batches
def process_file_in_chunks(input_file, chunk_size=1_000_000, workers=8):
    for chunk_idx, batch in enumerate(batch_iterator(read_smiles_file(input_file), chunk_size)):
        output_file = f"./data/zinc15_chunk_{chunk_idx:05d}.csv"
        print(f"Processing chunk {chunk_idx}, saving to {output_file}")

        # SKIP if file exists
        if os.path.exists(output_file):
            print(f"Skipping chunk {chunk_idx} (output exists: {output_file})")
            continue

        print(f"Processing chunk {chunk_idx}, saving to {output_file}")

        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['smiles', 'zinc_id', 'selfies'])

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_record = {executor.submit(process_smiles, record): record for record in batch}
                for future in tqdm(as_completed(future_to_record), total=len(batch), desc=f"Chunk {chunk_idx}"):
                    result = future.result()
                    if result:
                        writer.writerow(result)

process_file_in_chunks("zinc15_all_raw.smi", chunk_size=1_000_000, workers=24)

