import multiprocessing as mp
import os

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, LogitsProcessorList, \
    MinLengthLogitsProcessor

from token_mapping import token_mapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 🧬 Map tokens in generated SELFIES strings using token_mapping
def apply_selfies_token_mapping(selfies_str):
    """
    Applies post-processing token mapping for SELFIES
    Args:
        selfies_str: string with SELFIES tokens.

    Returns:

    """
    tokens = sf.split_selfies(selfies_str)
    mapped_tokens = [token_mapping.get(tok, tok) for tok in tokens]
    return "".join(mapped_tokens)


### --- 1️⃣ Convert SELFIES to SMILES --- ###
def selfies_to_smiles(selfies_str):
    """Convert a SELFIES string to a valid SMILES string."""
    try:
        smiles = sf.decoder(selfies_str)
        if Chem.MolFromSmiles(smiles):  # Ensure it's valid
            return smiles
    except Exception:
        return None
    return None


def canonicalize_smiles(smiles):
    """
    Standardize SMILES format for uniqueness checking.
    Args:
        smiles: SMILES strings

    Returns: string

    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


### --- 2️⃣ Load SMILES Data from CSV (Converted from SELFIES) --- ###
# TODO: 05/19/2025 21:15:00 maybe make this file already converted and combined...
# save the hassle of having to do this conversion EVERY time and in-memory...
def load_smiles_from_csv(csv_path, selfies_column="selfies"):
    """
    Load a set of SMILES from a CSV file that contains SELFIES.

    Parameters:
    - csv_path: Path to the CSV file
    - selfies_column: Column name containing SELFIES strings

    Returns:
    - A set of unique SMILES
    """
    try:
        df = pd.read_csv(csv_path, usecols=[selfies_column])  # Load only the SELFIES column
        df["SMILES"] = df[selfies_column].astype(str).apply(selfies_to_smiles)  # Convert SELFIES to SMILES
        smiles_set = set(df["SMILES"].dropna())  # Drop invalid conversions & convert to set
        print(f"✅ Loaded {len(smiles_set):,} unique SMILES from {csv_path} (converted from SELFIES)")
        return smiles_set
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return set()


# def load_selfies_from_df(df, selfies_column="selfies"):
#     """
#     Load a set of SMILES from a CSV file that contains SELFIES.
#
#     Parameters:
#     - df: Pandas DataFrame containing SELFIES column
#     - selfies_column: Column name containing SELFIES strings
#
#     Returns:
#     - A set of unique SMILES
#     """
#     if 'selfies' in df.columns:
#         continue
#     try:
#         # df = pd.read_csv(csv_path, usecols=[selfies_column])  # Load only the SELFIES column
#         # I should keep selfies from the original molecule grabbing...
#         # df["SMILES"] = df[selfies_column].astype(str).apply(selfies_to_smiles)  # Convert SELFIES to SMILES
#         # smiles_set = set(df["SMILES"].dropna())  # Drop invalid conversions & convert to set
#         # print(f"✅ Loaded {len(smiles_set):,} unique SMILES from {csv_path} (converted from SELFIES)")
#         # return smiles_set
#     # except Exception as e:
#     #     print(f"❌ Error loading {csv_path}: {e}")
    #     return set()


### --- 3️⃣ Define Model Wrapper for Generation --- ###
class HuggingFaceMoleculeGenerator:
    def __init__(self, model_path, tokenizer_path, device="cuda"):
        """Load Hugging Face BART model & tokenizer for molecule generation."""
        self.device = torch.device(device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # ✅ Load model and weights
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Fine-tuned model loaded from {model_path}")

    def selfies_to_smiles(self, selfies_list):
        """Convert SELFIES to valid SMILES."""
        smiles_list = []
        for selfies in selfies_list:
            try:
                smiles = sf.decoder(selfies)
                if Chem.MolFromSmiles(smiles):  # Ensure valid molecule
                    smiles_list.append(smiles)
            except Exception:
                continue  # Skip invalid molecules
        return smiles_list

    def sample(self, n, batch_size=100, prefix=""):
        """
        Generate `n` molecules in smaller batches and return SMILES strings.
        Args:
            n: Number of molecules to sample
            batch_size: Size of batch
            prefix: Molecule starting tokens in SELFIES (tokenizer vocabulary)

        Returns:

        """
        if not prefix.strip():
            print("⚠️ Empty prefix provided. Using default '<s>'.")
            prefix = "<s>"

        # input_text = "[C][N]"  # Example SELFIES input # DOESN'T DO A DIRECT STARTING BECAUSE OF THE BART MODEL STRUCTURE... 03-07-2025 TODO: ELABORATE ON THIS!
        input_text = prefix
        # tokens = input_text.split()
        tokens = list(sf.split_selfies(input_text))
        print(tokens)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(self.device)
        print(input_ids)
        # input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        # forced_bos_tokens = self.tokenizer("[C][N]", return_tensors="pt").input_ids[0].tolist()
        forced_bos_tokens = input_ids[0].tolist()

        # Helper to force the prefix during generation
        def prefix_allowed_fn(batch_id, input_ids):
            # First few tokens must match forced prefix
            prefix_len = len(forced_bos_tokens)
            if len(input_ids) < prefix_len:
                return [forced_bos_tokens[len(input_ids)]]
            return list(range(self.tokenizer.vocab_size))  # After prefix, allow all tokens

        generated_smiles = []

        for _ in range(0, n, batch_size):
            current_batch_size = min(batch_size, n - len(generated_smiles))  # Handle last batch

            with torch.no_grad():
                # output_ids = self.model.generate(
                #     input_ids.expand(current_batch_size, -1),  # Duplicate input for batch processing
                #     max_length=500,
                #     num_return_sequences=1,
                #     do_sample=True,
                #     temperature=2.5,
                #     top_k=50,
                #     top_p=0.95,
                #     repetition_penalty=2.3,
                #     num_beams=1
                # )
                # output_ids = self.model.generate(
                #     input_ids=input_ids.expand(batch_size, -1),
                #     max_length=50,
                #     do_sample=True,
                #     temperature=1.2,
                #     top_k=50,
                #     top_p=0.95,
                #     repetition_penalty=1.2,
                #     num_beams=1
                #     #         prefix_allowed_tokens_fn=prefix_allowed_fn
                # )
                output_ids = self.model.generate(
                    input_ids=input_ids.expand(batch_size, -1),
                    max_length=50,
                    do_sample=True,
                    temperature=1.2,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    num_beams=1,
                    prefix_allowed_tokens_fn=prefix_allowed_fn
                )

            print(f"Generated token IDs:\n{output_ids}")

            selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print(f"Decoded outputs:\n{selfies_list}")

            # print("🔎 Raw SELFIES:")
            # print(selfies_list[:10])

            # Apply mapping to all generated selfies before decoding
            mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
            print("\n🧬 Mapped SELFIES after token mapping:")
            print(mapped_selfies_list[:10])

            # smiles_list = self.selfies_to_smiles(selfies_list)
            smiles_list = self.selfies_to_smiles(mapped_selfies_list)

            generated_smiles.extend([s for s in smiles_list if s is not None])  # Filter out invalid molecules

            print("Cleaned SELFIES:")
            print(generated_smiles)

        return generated_smiles
        # return mapped_selfies_list


### --- 4️⃣ Benchmarking: Compute Uniqueness & Novelty and Save to CSV --- ###
def is_valid_smiles(smiles):
    """
    Check if a SMILES string is valid using RDKit.
    Args:
        smiles:

    Returns:
        Any:
    """
    return Chem.MolFromSmiles(smiles) is not None


def check_novelty(smiles_list, train_set, pretrain_set):
    """Parallelized function to check novelty of molecules.

    Args:
        smiles_list:
        train_set:
        pretrain_set:

    Returns:
        list[Any]:
    """
    return [smiles for smiles in smiles_list if smiles not in train_set and smiles not in pretrain_set]


def benchmark_generated_molecules_selfies(gen, selfies_df, num_samples=5000, batch_size=100,
                                          output_csv="generated_molecules.csv"):
    """
    Parameters:
    - gen: HuggingFaceMoleculeGenerator instance
    - selfies_df: Pandas DataFrame with columns 'SELFIES' and 'SMILES'
    - num_samples: Total number of molecules to generate
    - batch_size: Number of molecules generated per batch
    - output_csv: Path to save generated molecules

    Returns:
    - A dictionary with uniqueness and novelty scores
    """

    # ✅ Extract training SELFIES column as a set for fast lookup
    # train_selfies_set = set(selfies_df["selfies"].dropna().astype(str))
    train_selfies_set = set(selfies_df["selfies"].dropna().astype(str).unique())

    # ✅ Generate SELFIES strings
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="[C]")

    # ✅ Filter out empty or non-string entries
    filtered_selfies = [s for s in generated_selfies if isinstance(s, str) and s.strip()]
    print(f"✅ Valid (non-empty) SELFIES: {len(filtered_selfies)} / {len(generated_selfies)}")

    # ✅ Uniqueness: how many are unique among generated
    unique_selfies_set = set(filtered_selfies)
    uniqueness = (len(unique_selfies_set) / len(filtered_selfies)) * 100 if filtered_selfies else 0

    # ✅ Novelty: how many are not in training data
    novel_selfies = list(unique_selfies_set - train_selfies_set)
    novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0

    # ✅ Save to CSV
    df = pd.DataFrame({
        "SELFIES": list(unique_selfies_set),
        "Novel": [s in novel_selfies for s in unique_selfies_set],
        "Unique": [True] * len(unique_selfies_set)
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # ✅ Summary results
    results = {
        "Total Generated": len(generated_selfies),
        "Valid Unique SELFIES": len(unique_selfies_set),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_selfies),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results

def benchmark_generated_molecules_smiles(gen, selfies_df, num_samples=5000, batch_size=100,
                                         output_csv="generated_molecules.csv"):
    """
    Benchmarks molecules generated by a HuggingFaceMoleculeGenerator.

    Parameters:
    - gen: HuggingFaceMoleculeGenerator instance
    - selfies_df: Pandas DataFrame with 'SMILES' and 'SELFIES' columns
    - num_samples: Total number of molecules to generate
    - batch_size: Number of molecules generated per batch
    - output_csv: Path to save generated molecules

    Returns:
    - Dictionary with uniqueness and novelty metrics
    """

    # ✅ Reference: Canonicalized SMILES from training data
    train_smiles_set = set(selfies_df["smiles"].dropna().astype(str).unique())

    # ✅ Generate molecules (SMILES)
    generated_smiles = gen.sample(num_samples, batch_size=batch_size, prefix="[C]")

    # ✅ Filter out invalid/empty SMILES
    valid_smiles = [s for s in generated_smiles if isinstance(s, str) and s.strip()]
    print(f"✅ Valid SMILES: {len(valid_smiles)} / {len(generated_smiles)}")

    # ✅ Uniqueness: unique molecules within generated set
    unique_smiles_set = set(valid_smiles)
    uniqueness = (len(unique_smiles_set) / len(valid_smiles)) * 100 if valid_smiles else 0

    # ✅ Novelty: molecules not in training data
    novel_smiles = list(unique_smiles_set - train_smiles_set)
    novelty = (len(novel_smiles) / len(unique_smiles_set)) * 100 if unique_smiles_set else 0

    # ✅ Save results to CSV
    df = pd.DataFrame({
        "SMILES": list(unique_smiles_set),
        "Novel": [s in novel_smiles for s in unique_smiles_set],
        "Unique": [True] * len(unique_smiles_set)
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # ✅ Summary output
    results = {
        "Total Generated": len(generated_smiles),
        "Valid Unique SMILES": len(unique_smiles_set),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_smiles),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    for key, val in results.items():
        print(f"{key}: {val}")

    return results


### --- 5️⃣ Run Benchmarking --- ###
if __name__ == "__main__":
    import time

    # num_samples = 1_000_000
    # batch_size = 1_000

    num_samples = 25
    batch_size = 1

    output_csv = f"generated_molecules_num_samples_{str(num_samples)}_batch_{batch_size}.csv"
    # TODO: 06/22/2025 GET full dataset from singular csv
    # This results in the df with the full pass...
    # load_selfies_from_csv should be similar to load_smiles_from_csv BUT
    # check to see if df has the ['selfies'] column and if so it is good
    # if no ['selfies'] apply the function
    # SMILES already should exist.... ugh... going backwards?
    # This should really be checking for unique SELFIES and not SMILES?
    # Generation is in SELFIES...

    # pretrain_smiles = load_smiles_from_csv("data/trainable_selfies_model_nada.csv")
    # pretrain_smiles = load_smiles_from_csv("data/molecule_data_model_7_adafactor_invsqrt.csv")
    # TODO: 05/19/2025 16:33 pm get this finetune list of molecules going... Maybe same DF as that used in training the tokenizer?
    # finetune_smiles = load_smiles_from_csv("data/smiles_finetune_data_properties_selfies.csv")

    selfies_df = pd.read_csv("combined_selfies_dataset_benchmarkgen.csv")


    gen = HuggingFaceMoleculeGenerator(
        model_path="/home/kollin/Desktop/CollectedRuns_All_And_New/CollectedRuns_All_And_New/CollectedRuns/CLUSTER_RESULTS/extra/selfies_BART_PRETRAIN_model_4/model",
        tokenizer_path="selfies_word_tokenizer_12M",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    # benchmark_results = benchmark_generated_molecules(
    #     gen, finetune_smiles, pretrain_smiles, num_samples=100, batch_size=25, num_workers=12,
    #     output_csv="generated_molecules_100_new.csv"
    # )

    benchmark_results = benchmark_generated_molecules_smiles(
        gen, selfies_df, num_samples=num_samples, batch_size=batch_size, output_csv=output_csv
    )
    print(f"\n⏳ Benchmark completed in {round(time.time() - start_time, 2)} seconds.")
