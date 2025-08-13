import multiprocessing as mp
import os

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, LogitsProcessorList, \
    MinLengthLogitsProcessor

from token_mapping import token_mapping

import dask.dataframe as dd

import gc

from pybloom_live import ScalableBloomFilter


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_bloom_filter_from_parquet(parquet_path, column="selfies", error_rate=0.001):
    """
    Builds a Bloom filter from a large Parquet file.
    Returns a ScalableBloomFilter object.
    """
    print(f"📂 Loading training data from: {parquet_path}")
    df = dd.read_parquet(parquet_path, columns=[column])
    selfies_series = df[column].dropna().astype(str)

    print("🔄 Building Bloom filter...")
    bf = ScalableBloomFilter(initial_capacity=1_000_000_000, error_rate=error_rate)

    # Stream into Bloom filter
    for partition in selfies_series.to_delayed():
        part = partition.compute()
        for s in part:
            bf.add(s)

    print(f"✅ Bloom filter constructed with ~{len(bf)} entries.")
    return bf


def apply_selfies_token_mapping(selfies_str):
    """
    Applies post-processing token mapping for SELFIES
    Args:
        selfies_str: string with SELFIES tokens.

    Returns:

    """
    # Causing issues with "." character token because no "." split.
    tokens = selfies_str.split()  # Assumes space-delimited SELFIES tokens
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

    # def sample(self, n, batch_size=100, prefix=""):
    # def sample(self, n, batch_size=100, prefix="", output_csv=None):
    #     """
    #     Generate `n` molecules in smaller batches and return SMILES strings.
    #     Args:
    #         n: Number of molecules to sample
    #         batch_size: Size of batch
    #         prefix: Molecule starting tokens in SELFIES (tokenizer vocabulary)
    #
    #     Returns:
    #
    #     """
    #     if not prefix.strip():
    #         print("⚠️ Empty prefix provided. Using default '<s>'.")
    #         prefix = "<s>"
    #
    #     input_text = prefix
    #     tokens = input_text.split()  # Assumes space-delimited SELFIES tokens
    #     print(tokens)
    #     ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_ids = torch.tensor([ids]).to(self.device)
    #     print(input_ids)
    #
    #     generated_smiles = []
    #
    #     print(type(batch_size))
    #     # for _ in range(0, n, batch_size):
    #     #     current_batch_size = min(batch_size, n - len(generated_smiles))  # Handle last batch
    #     #
    #     #     with torch.no_grad():
    #     #         # output_ids = self.model.generate(
    #     #         #     input_ids.expand(current_batch_size, -1),  # Duplicate input for batch processing
    #     #         #     max_length=500,
    #     #         #     num_return_sequences=1,
    #     #         #     do_sample=True,
    #     #         #     temperature=2.5,
    #     #         #     top_k=50,
    #     #         #     top_p=0.95,
    #     #         #     repetition_penalty=2.3,
    #     #         #     num_beams=1
    #     #         # )
    #     #         # output_ids = self.model.generate(
    #     #         #     input_ids=input_ids.expand(batch_size, -1),
    #     #         #     max_length=50,
    #     #         #     do_sample=True,
    #     #         #     temperature=1.2,
    #     #         #     top_k=50,
    #     #         #     top_p=0.95,
    #     #         #     repetition_penalty=1.2,
    #     #         #     num_beams=1
    #     #         #     #         prefix_allowed_tokens_fn=prefix_allowed_fn
    #     #         # )
    #     #         # output_ids = self.model.generate(
    #     #         #     input_ids=input_ids.expand(batch_size, -1),
    #     #         #     max_length=22,
    #     #         #     num_return_sequences=batch_size,
    #     #         #     do_sample=True,
    #     #         #     temperature=1.2,
    #     #         #     top_k=50,
    #     #         #     top_p=0.95,
    #     #         #     repetition_penalty=1.2,
    #     #         #     num_beams=batch_size
    #     #         #     # prefix_allowed_tokens_fn=prefix_allowed_fn
    #     #         # )
    #     #         # output_ids = self.model.generate(
    #     #         #     input_ids=input_ids.expand(batch_size, -1),
    #     #         #     max_length=22,
    #     #         #     num_return_sequences=batch_size,
    #     #         #     do_sample=True,
    #     #         #     temperature=2.5,
    #     #         #     top_k=50,
    #     #         #     top_p=0.95,
    #     #         #     repetition_penalty=1.2,
    #     #         #     num_beams=batch_size
    #     #         # )
    #     #         # output_ids = self.model.generate(
    #     #         #     input_ids=input_ids.repeat(batch_size, 1),
    #     #         #     max_length=22,
    #     #         #     num_return_sequences=batch_size,
    #     #         #     do_sample=True,
    #     #         #     temperature=2.5,
    #     #         #     top_k=50,
    #     #         #     top_p=0.95,
    #     #         #     repetition_penalty=1.2,
    #     #         #     num_beams=batch_size
    #     #         # )
    #     #         output_ids = self.model.generate(
    #     #             input_ids=input_ids.expand(batch_size, -1),
    #     #             max_length=22,
    #     #             num_return_sequences=batch_size,
    #     #             do_sample=True,
    #     #             temperature=2.5,
    #     #             top_k=50,
    #     #             top_p=0.95,
    #     #             repetition_penalty=1.2
    #     #         )
    #     #
    #     #
    #     #     print(f"Generated token IDs:\n{output_ids}")
    #     #
    #     #     selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     #     # decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     #     print(f"Decoded outputs:\n{selfies_list}")
    #     #
    #     #     # Apply mapping to all generated selfies before decoding
    #     #     mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
    #     #     print("\n🧬 Mapped SELFIES after token mapping:")
    #     #     print(mapped_selfies_list[:10])
    #     #     # generated_smiles.extend(mapped_selfies_list)
    #     #
    #     #     smiles_list = []
    #     #     for molecules in mapped_selfies_list:
    #     #         smiles_list.append(molecules.replace(" ", ""))
    #     #     # generated_smiles.extend([s for s in smiles_list if s is not None])  # Filter out invalid molecules
    #     #     with open(output_csv, "a") as f:
    #     #         for s in mapped_selfies_list:
    #     #             if s:
    #     #                 f.write(f"{s},True,Novel\n")  # use actual novelty check if needed
    #
    #     total_generated = 0  # Initialize before loop
    #
    #     for _ in range(0, n, batch_size):
    #         current_batch_size = min(batch_size, n - total_generated)
    #
    #         with torch.no_grad():
    #             output_ids = self.model.generate(
    #                 input_ids=input_ids.expand(current_batch_size, -1),
    #                 max_length=22,
    #                 num_return_sequences=current_batch_size,
    #                 do_sample=True,
    #                 temperature=1.5,
    #                 top_k=30,
    #                 top_p=0.9,
    #                 repetition_penalty=1.1,
    #             )
    #
    #         output_ids = output_ids.cpu()
    #         selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #         # generated_smiles.extend(mapped_selfies_list)
    #         mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
    #
    #         with open(output_csv, "a") as f:
    #             for s in mapped_selfies_list:
    #                 if s:
    #                     f.write(f"{s},True,Novel\n")
    #
    #         del output_ids
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #
    #         print("Cleaned SELFIES:")
    #         print(generated_smiles)
    #
    #         total_generated += current_batch_size
    #
    #     return generated_smiles
    #     # return mapped_selfies_list


    # def sample(self, n, batch_size=100, prefix="", output_csv="output2.csv"):
    #     """
    #     Generate `n` molecules in batches and return SMILES strings.
    #
    #     Args:
    #         n (int): Total number of molecules to generate.
    #         batch_size (int): Number of molecules per batch.
    #         prefix (str): Starting SELFIES tokens (space-separated).
    #         output_csv (str or None): Path to write outputs (optional).
    #
    #     Returns:
    #         List[str]: List of generated SMILES strings.
    #     """
    #     if not prefix.strip():
    #         print("⚠️ Empty prefix provided. Using default '<s>'.")
    #         prefix = "<s>"
    #
    #     tokens = prefix.strip().split()
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_tensor = torch.tensor([input_ids], device=self.device)
    #
    #     total_generated = 0
    #     all_smiles = []
    #
    #     while total_generated < n:
    #         current_batch_size = min(batch_size, n - total_generated)
    #
    #         with torch.no_grad():
    #             output_ids = self.model.generate(
    #                 input_ids=input_tensor.expand(current_batch_size, -1),
    #                 max_length=22,
    #                 num_return_sequences=current_batch_size,
    #                 do_sample=True,
    #                 temperature=1.5,
    #                 top_k=30,
    #                 top_p=0.9,
    #                 repetition_penalty=1.1,
    #             )
    #
    #         selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #
    #         # ✅ Apply token cleanup if needed
    #         mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
    #
    #         # ✅ Convert SELFIES to valid SMILES
    #         smiles_list = self.selfies_to_smiles(mapped_selfies_list)
    #         all_smiles.extend(smiles_list)
    #
    #         # ✅ Save to file if requested
    #         if output_csv:
    #             with open(output_csv, "a") as f:
    #                 for s in smiles_list:
    #                     f.write(f"{s},True,Novel\n")
    #
    #         total_generated += current_batch_size
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #
    #     return all_smiles

    # def sample(self, n, batch_size=100, prefix="", output_csv="output2.csv"):
    #     """
    #     Generate `n` molecules in batches and return SMILES strings.
    #
    #     Args:
    #         n (int): Total number of molecules to generate.
    #         batch_size (int): Number of molecules per batch.
    #         prefix (str): Starting SELFIES tokens (space-separated).
    #         output_csv (str or None): Path to write outputs (optional).
    #
    #     Returns:
    #         List[str]: List of generated SMILES strings.
    #     """
    #     if not prefix.strip():
    #         print("⚠️ Empty prefix provided. Using default '<s>'.")
    #         prefix = "<s>"
    #
    #     tokens = prefix.strip().split()
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_tensor = torch.tensor([input_ids], device=self.device)
    #
    #     total_generated = 0
    #     all_selfies = []
    #
    #     while total_generated < n:
    #         current_batch_size = min(batch_size, n - total_generated)
    #
    #         with torch.no_grad():
    #             output_ids = self.model.generate(
    #                 input_ids=input_tensor.expand(current_batch_size, -1),
    #                 max_length=22,
    #                 num_return_sequences=current_batch_size,
    #                 do_sample=True,
    #                 temperature=1.5,
    #                 top_k=30,
    #                 top_p=0.9,
    #                 repetition_penalty=1.1,
    #             )
    #
    #         selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #
    #         # ✅ Apply token cleanup if needed
    #         mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
    #
    #         # ✅ Convert SELFIES to valid SMILES
    #         # smiles_list = self.selfies_to_smiles(mapped_selfies_list)
    #         # all_smiles.extend(smiles_list)
    #
    #         all_selfies.extend(selfies_list)
    #         # ✅ Save to file if requested
    #         if output_csv:
    #             with open(output_csv, "a") as f:
    #                 for s in mapped_selfies_list:
    #                     f.write(f"{s},True,Novel\n")
    #
    #         total_generated += current_batch_size
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #
    #     return all_selfies
    #
    #     # return all_smiles

    def sample(self, n, batch_size=100, prefix="", output_csv="output2.csv"):
        """
        Generate `n` molecules in batches and return SMILES strings.

        Args:
            n (int): Total number of molecules to generate.
            batch_size (int): Number of molecules per batch.
            prefix (str): Starting SELFIES tokens (space-separated).
            output_csv (str or None): Path to write outputs (optional).

        Returns:
            List[str]: List of generated SMILES strings.
        """
        if not prefix.strip():
            print("⚠️ Empty prefix provided. Using default '<s>'.")
            prefix = "<s>"

        tokens = prefix.strip().split()
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids], device=self.device)

        total_generated = 0
        all_selfies = []

        while total_generated < n:
            current_batch_size = min(batch_size, n - total_generated)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_tensor.expand(current_batch_size, -1),
                    max_length=63,
                    num_return_sequences=current_batch_size,
                    do_sample=True,
                    temperature=1.5,
                    top_k=30,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # ✅ Apply token cleanup if needed
            # mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]

            # ✅ Convert SELFIES to valid SMILES
            # smiles_list = self.selfies_to_smiles(mapped_selfies_list)
            # all_smiles.extend(smiles_list)


            # all_selfies_list = []
            # for molecules in all_selfies:
            #     all_selfies.append(molecules.replace(" ", ""))

            all_selfies_list = []
            for molecules in selfies_list:
                all_selfies_list.append(molecules.replace(" ", ""))

            # all_selfies.extend(selfies_list)
            all_selfies.extend(all_selfies_list)

            # TODO: 08/01/25 16:45 pm: ALSO CONSIDER DOING MAPPED SELFIES HERE!

            # # ✅ Save to file if requested
            # if output_csv:
            #     with open(output_csv, "a") as f:
            #         for s in selfies_list:
            #             f.write(f"{s},True,Novel\n")

            # ✅ Save to file if requested
            if output_csv:
                with open(output_csv, "a") as f:
                    for s in all_selfies:
                        f.write(f"{s},True,Novel\n")

            total_generated += current_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        return all_selfies

        # return all_smiles


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
                                         output_csv="generated_molecules.csv", output_txt="generated_molecules.txt"):
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
    generated_smiles = gen.sample(num_samples, batch_size=batch_size, prefix="")

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
        "SELFIES": list(unique_smiles_set), # changed column header to SELFIES instead of SMILES.
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
    with open(output_txt, "w") as f:
        f.write("📊 Benchmark Results:\n")
        for key, val in results.items():
            line = f"{key}: {val}"
            print(line)
            f.write(line + "\n")

    print(f"\n📝 Benchmark summary saved to: {output_txt}")

    return results

import pandas as pd
import os

# def benchmark_generated_molecules_selfies_parquet(gen, selfies_pt, num_samples=5000, batch_size=100,
#                                                   output_csv="generated_molecules.csv",
#                                                   output_txt="generated_molecules.txt"):
#     """
#     Benchmarks molecules generated by a HuggingFaceMoleculeGenerator.
#
#     Parameters:
#     - gen: HuggingFaceMoleculeGenerator instance
#     - selfies_pt: Either a DataFrame or path to a Parquet file with column 'smiles'
#     - num_samples: Total number of molecules to generate
#     - batch_size: Number of molecules generated per batch
#     - output_csv: Path to save generated molecules
#     - output_txt: Path to save benchmark summary
#
#     Returns:
#     - Dictionary with uniqueness and novelty metrics
#     """
#
#     # ✅ Load Parquet file if path is given
#     if isinstance(selfies_pt, str):
#         if not os.path.exists(selfies_pt):
#             raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
#         print(f"📂 Loading training data from: {selfies_pt}")
#         selfies_df = dd.read_parquet(selfies_pt)
#     elif isinstance(selfies_pt, pd.DataFrame):
#         selfies_df = selfies_pt
#     else:
#         raise ValueError("`selfies_pt` must be a path to a Parquet file or a pandas DataFrame.")
#
#     # ✅ Reference set of SMILES from training data
#     # if "smiles" not in selfies_df.columns:
#     #     raise ValueError("Training dataset must contain a 'smiles' column.")
#     # train_smiles_set = set(selfies_df["smiles"].dropna().astype(str).unique())
#
#     # train_selfies_set = set(selfies_df["selfies"].dropna().astype(str).unique())
#     # train_selfies_set = set(
#     #     selfies_df["selfies"]
#     #     .dropna()
#     #     .astype(str)
#     #     .unique()
#     #     .compute()
#     # )
#
#     # # Extract unique values in each partition, then deduplicate
#     # train_selfies_set = set(
#     #     selfies_df["selfies"]
#     #     .dropna()
#     #     .astype(str)
#     #     .map_partitions(lambda df: df.unique())
#     #     .compute()
#     #     .unique()
#     # )
#
#     # train_selfies_series = selfies_df["selfies"].dropna().astype(str).drop_duplicates().compute()
#     # train_selfies_set = set(train_selfies_series)
#
#     train_selfies_set = set(
#         selfies_df["selfies"]
#         .dropna()
#         .astype(str)
#         .map_partitions(lambda df: df.drop_duplicates())
#         .compute()
#         .drop_duplicates()
#     )
#
#     # ✅ Generate molecules (SMILES)
#     generated_smiles = gen.sample(num_samples, batch_size=batch_size, prefix="")
#
#     # ✅ Filter out invalid/empty SMILES
#     valid_smiles = [s for s in generated_smiles if isinstance(s, str) and s.strip()]
#     print(f"✅ Valid SMILES: {len(valid_smiles)} / {len(generated_smiles)}")
#
#     # ✅ Uniqueness: unique molecules within generated set
#     unique_smiles_set = set(valid_smiles)
#     uniqueness = (len(unique_smiles_set) / len(valid_smiles)) * 100 if valid_smiles else 0
#
#     # ✅ Novelty: molecules not in training data
#     novel_smiles = list(unique_smiles_set - train_selfies_set)
#     novelty = (len(novel_smiles) / len(unique_smiles_set)) * 100 if unique_smiles_set else 0
#
#     # ✅ Save results to CSV
#     df = pd.DataFrame({
#         "SELFIES": list(unique_smiles_set),  # Column name kept as "SELFIES" per your design
#         "Novel": [s in novel_smiles for s in unique_smiles_set],
#         "Unique": [True] * len(unique_smiles_set)
#     })
#     df.to_csv(output_csv, index=False)
#     print(f"📁 Generated molecules saved to: {output_csv}")
#
#     # ✅ Summary output
#     results = {
#         "Total Generated": len(generated_smiles),
#         "Valid Unique SMILES": len(unique_smiles_set),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel Molecules": len(novel_smiles),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     with open(output_txt, "w") as f:
#         f.write("📊 Benchmark Results:\n")
#         for key, val in results.items():
#             line = f"{key}: {val}"
#             print(line)
#             f.write(line + "\n")
#
#     print(f"\n📝 Benchmark summary saved to: {output_txt}")
#
#     return results
#

# def benchmark_generated_selfies_parquet(gen, selfies_pt, num_samples=5000, batch_size=100,
#                                        output_csv="generated_molecules.csv",
#                                        output_txt="generated_molecules.txt"):
#     """
#     Benchmarks SELFIES generated by a HuggingFaceMoleculeGenerator.
#
#     Parameters:
#     - gen: HuggingFaceMoleculeGenerator instance
#     - selfies_pt: Either a DataFrame or path to a Parquet file with column 'selfies'
#     - num_samples: Total number of SELFIES to generate
#     - batch_size: Number of SELFIES generated per batch
#     - output_csv: Path to save generated molecules
#     - output_txt: Path to save benchmark summary
#
#     Returns:
#     - Dictionary with uniqueness and novelty metrics
#     """
#
#     import os
#     import pandas as pd
#     import dask.dataframe as dd
#
#     # Load training SELFIES data
#     if isinstance(selfies_pt, str):
#         if not os.path.exists(selfies_pt):
#             raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
#         print(f"📂 Loading training data from: {selfies_pt}")
#         selfies_df = dd.read_parquet(selfies_pt)
#     elif isinstance(selfies_pt, pd.DataFrame):
#         selfies_df = selfies_pt
#     else:
#         raise ValueError("`selfies_pt` must be a path to a Parquet file or a pandas DataFrame.")
#
#     # Get unique SELFIES from training data
#     train_selfies_set = set(
#         selfies_df["selfies"]
#         .dropna()
#         .astype(str)
#         .map_partitions(lambda df: df.drop_duplicates())
#         .compute()
#         .drop_duplicates()
#     )
#
#     # Generate SELFIES molecules
#     generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")
#
#     # Filter out empty or None entries
#     filtered_selfies = [s for s in generated_selfies if isinstance(s, str) and s.strip()]
#     print(f"✅ Non-empty SELFIES: {len(filtered_selfies)} / {len(generated_selfies)}")
#
#     # Uniqueness in generated SELFIES
#     unique_selfies_set = set(filtered_selfies)
#     uniqueness = (len(unique_selfies_set) / len(filtered_selfies)) * 100 if filtered_selfies else 0
#
#     # Novelty compared to training SELFIES
#     novel_selfies = list(unique_selfies_set - train_selfies_set)
#     novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0
#
#     # Save results to CSV
#     df = pd.DataFrame({
#         "SELFIES": list(unique_selfies_set),
#         "Novel": [s in novel_selfies for s in unique_selfies_set],
#         "Unique": [True] * len(unique_selfies_set)
#     })
#     df.to_csv(output_csv, index=False)
#     print(f"📁 Generated SELFIES saved to: {output_csv}")
#
#     # Summary output
#     results = {
#         "Total Generated": len(generated_selfies),
#         "Non-empty SELFIES": len(filtered_selfies),
#         "Unique SELFIES": len(unique_selfies_set),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel SELFIES": len(novel_selfies),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     with open(output_txt, "w") as f:
#         f.write("📊 Benchmark Results:\n")
#         for key, val in results.items():
#             line = f"{key}: {val}"
#             print(line)
#             f.write(line + "\n")
#
#     print(f"\n📝 Benchmark summary saved to: {output_txt}")
#
#     return results

import os
import pandas as pd
import dask.dataframe as dd

# def benchmark_generated_molecules_selfies_parquet(
#     gen, selfies_pt, num_samples=5000, batch_size=100,
#     output_csv="generated_molecules.csv",
#     output_txt="generated_molecules.txt"
# ):
#     """
#     Benchmarks molecules generated by a HuggingFaceMoleculeGenerator.
#
#     Parameters:
#     - gen: HuggingFaceMoleculeGenerator instance
#     - selfies_pt: Either a DataFrame or path to a Parquet file with column 'selfies'
#     - num_samples: Total number of molecules to generate
#     - batch_size: Number of molecules generated per batch
#     - output_csv: Path to save generated molecules
#     - output_txt: Path to save benchmark summary
#
#     Returns:
#     - Dictionary with uniqueness and novelty metrics
#     """
#
#     # Load Parquet file if path is given
#     if isinstance(selfies_pt, str):
#         if not os.path.exists(selfies_pt):
#             raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
#         print(f"📂 Loading training data from: {selfies_pt}")
#         selfies_df = dd.read_parquet(selfies_pt)
#     elif isinstance(selfies_pt, pd.DataFrame):
#         selfies_df = selfies_pt
#     else:
#         raise ValueError("`selfies_pt` must be a path to a Parquet file or a pandas DataFrame.")
#
#     # Build training set of SELFIES memory-efficiently, partition by partition
#     train_selfies_set = set()
#     print("🔄 Computing unique SELFIES from training data partition-wise...")
#     for delayed_partition in selfies_df["selfies"].dropna().astype(str).to_delayed():
#         partition_df = delayed_partition.compute()
#         unique_selfies = set(partition_df.drop_duplicates())
#         train_selfies_set.update(unique_selfies)
#     print(f"✅ Training SELFIES count: {len(train_selfies_set)}")
#
#     # Generate molecules (SELFIES)
#     generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")
#
#     # Filter out empty or invalid generated SELFIES (if needed)
#     # Assuming generated_selfies are all strings and valid, so no filtering here
#
#     # Uniqueness: unique molecules within generated set
#     unique_selfies_set = set(generated_selfies)
#     uniqueness = (len(unique_selfies_set) / len(generated_selfies)) * 100 if generated_selfies else 0
#
#     # Novelty: molecules not in training data
#     novel_selfies = list(unique_selfies_set - train_selfies_set)
#     novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0
#
#     # Save results to CSV
#     df = pd.DataFrame({
#         "SELFIES": list(unique_selfies_set),
#         "Novel": [s in novel_selfies for s in unique_selfies_set],
#         "Unique": [True] * len(unique_selfies_set)
#     })
#     df.to_csv(output_csv, index=False)
#     print(f"📁 Generated molecules saved to: {output_csv}")
#
#     # Summary output
#     results = {
#         "Total Generated": len(generated_selfies),
#         "Valid Unique SELFIES": len(unique_selfies_set),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel Molecules": len(novel_selfies),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     with open(output_txt, "w") as f:
#         f.write("📊 Benchmark Results:\n")
#         for key, val in results.items():
#             line = f"{key}: {val}"
#             print(line)
#             f.write(line + "\n")
#
#     print(f"\n📝 Benchmark summary saved to: {output_txt}")
#
#     return results

import os
import pandas as pd
import dask.dataframe as dd

def benchmark_generated_molecules_selfies_parquet(
    gen, selfies_pt, num_samples=5000, batch_size=100,
    output_csv="generated_molecules.csv",
    output_txt="generated_molecules.txt",
    temp_train_set_path="train_selfies.txt"
):
    """
    Benchmarks generated SELFIES for uniqueness and novelty.

    Parameters:
    - gen: HuggingFaceMoleculeGenerator
    - selfies_pt: Path to a Parquet file or a DataFrame with a 'selfies' column
    - num_samples: Number of SELFIES to generate
    - batch_size: Generation batch size
    - output_csv: Path to save generated molecules
    - output_txt: Path to save benchmark summary
    - temp_train_set_path: Path to cache deduplicated training SELFIES
    """

    # Step 1: Save training SELFIES to disk (if not already done)
    if not os.path.exists(temp_train_set_path):
        print(f"📂 Writing unique training SELFIES to: {temp_train_set_path}")

        if isinstance(selfies_pt, str):
            selfies_df = dd.read_parquet(selfies_pt)
        elif isinstance(selfies_pt, pd.DataFrame):
            selfies_df = dd.from_pandas(selfies_pt, npartitions=8)
        else:
            raise ValueError("`selfies_pt` must be a path or a pandas DataFrame.")

        selfies_df["selfies"] = selfies_df["selfies"].astype(str)
        selfies_df["selfies"].dropna().drop_duplicates().to_csv(
            temp_train_set_path,
            single_file=True,
            index=False,
            header=False
        )

    # Step 2: Load training SELFIES from disk (now small)
    with open(temp_train_set_path, "r") as f:
        train_selfies_set = set(line.strip() for line in f if line.strip())
    print(f"✅ Loaded {len(train_selfies_set)} unique training SELFIES")

    # Step 3: Generate molecules (SELFIES)
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")

    # Step 4: Uniqueness
    unique_selfies_set = set(generated_selfies)
    uniqueness = (len(unique_selfies_set) / len(generated_selfies)) * 100 if generated_selfies else 0

    # Step 5: Novelty
    novel_selfies = list(unique_selfies_set - train_selfies_set)
    novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0

    # Step 6: Save output CSV
    df = pd.DataFrame({
        "SELFIES": list(unique_selfies_set),
        "Novel": [s in novel_selfies for s in unique_selfies_set],
        "Unique": [True] * len(unique_selfies_set)
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # Step 7: Save summary
    results = {
        "Total Generated": len(generated_selfies),
        "Valid Unique SELFIES": len(unique_selfies_set),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_selfies),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    with open(output_txt, "w") as f:
        f.write("📊 Benchmark Results:\n")
        for key, val in results.items():
            line = f"{key}: {val}"
            print(line)
            f.write(line + "\n")

    print(f"\n📝 Benchmark summary saved to: {output_txt}")
    return results

def benchmark_generated_molecules_selfies_parquet_bloom(
    gen,
    selfies_pt,
    num_samples=5000,
    batch_size=100,
    output_csv="generated_molecules.csv",
    output_txt="generated_molecules.txt",
    bloom_error_rate=0.001
):
    """
    Benchmarks generated SELFIES using a Bloom filter for novelty checking.
    """

    # ✅ Step 1: Load training data into Bloom filter
    if isinstance(selfies_pt, str):
        if not os.path.exists(selfies_pt):
            raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
        bloom_filter = build_bloom_filter_from_parquet(selfies_pt, error_rate=bloom_error_rate)
    elif isinstance(selfies_pt, pd.DataFrame):
        selfies_series = selfies_pt["selfies"].dropna().astype(str)
        bloom_filter = ScalableBloomFilter(initial_capacity=1_000_000_000, error_rate=bloom_error_rate)
        for s in selfies_series:
            bloom_filter.add(s)
    else:
        raise ValueError("`selfies_pt` must be a path to a Parquet file or a pandas DataFrame.")

    # ✅ Step 2: Generate SELFIES strings
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")

    # ✅ Step 3: Uniqueness
    unique_selfies_set = set(generated_selfies)
    uniqueness = (len(unique_selfies_set) / len(generated_selfies)) * 100 if generated_selfies else 0

    # ✅ Step 4: Novelty using Bloom filter
    novel_selfies = [s for s in unique_selfies_set if s not in bloom_filter]
    novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0

    # ✅ Step 5: Save to CSV
    df = pd.DataFrame({
        "SELFIES": list(unique_selfies_set),
        "Novel": [s in novel_selfies for s in unique_selfies_set],
        "Unique": [True] * len(unique_selfies_set),
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # ✅ Step 6: Save summary
    results = {
        "Total Generated": len(generated_selfies),
        "Valid Unique SELFIES": len(unique_selfies_set),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_selfies),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    with open(output_txt, "w") as f:
        f.write("📊 Benchmark Results:\n")
        for k, v in results.items():
            line = f"{k}: {v}"
            print(line)
            f.write(line + "\n")

    print(f"\n📝 Benchmark summary saved to: {output_txt}")
    return results



import os
import pandas as pd
from cuckoopy import CuckooFilter

def benchmark_generated_molecules_selfies_parquet_cuckoo(
    gen,
    selfies_pt,
    num_samples=5000,
    batch_size=100,
    output_csv="generated_molecules.csv",
    output_txt="generated_molecules.txt",
    cuckoo_capacity=900_000_000,
    bucket_size=4,
    fingerprint_size=16,
    max_displacements=500
):
    """
    Benchmarks generated SELFIES using a Cuckoo filter for novelty checking.
    """

    # ✅ Step 1: Load training data into Cuckoo filter
    cuckoo_filter = CuckooFilter(
        capacity=cuckoo_capacity,
        bucket_size=bucket_size,
        fingerprint_size=fingerprint_size,
        max_displacements=max_displacements
    )

    if isinstance(selfies_pt, str):
        if not os.path.exists(selfies_pt):
            raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
        df_train = pd.read_parquet(selfies_pt)
        selfies_series = df_train["selfies"].dropna().astype(str)
    elif isinstance(selfies_pt, pd.DataFrame):
        selfies_series = selfies_pt["selfies"].dropna().astype(str)
    else:
        raise ValueError("`selfies_pt` must be a path to a Parquet file or a pandas DataFrame.")

    for s in selfies_series:
        cuckoo_filter.insert(s)

    # ✅ Step 2: Generate SELFIES strings
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")

    # ✅ Step 3: Uniqueness
    unique_selfies_set = set(generated_selfies)
    uniqueness = (len(unique_selfies_set) / len(generated_selfies)) * 100 if generated_selfies else 0

    # ✅ Step 4: Novelty using Cuckoo filter
    novel_selfies = [s for s in unique_selfies_set if not cuckoo_filter.contains(s)]
    novelty = (len(novel_selfies) / len(unique_selfies_set)) * 100 if unique_selfies_set else 0

    # ✅ Step 5: Save to CSV
    df = pd.DataFrame({
        "SELFIES": list(unique_selfies_set),
        "Novel": [s in novel_selfies for s in unique_selfies_set],
        "Unique": [True] * len(unique_selfies_set),
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # ✅ Step 6: Save summary
    results = {
        "Total Generated": len(generated_selfies),
        "Valid Unique SELFIES": len(unique_selfies_set),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_selfies),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    with open(output_txt, "w") as f:
        f.write("📊 Benchmark Results:\n")
        for k, v in results.items():
            line = f"{k}: {v}"
            print(line)
            f.write(line + "\n")

    print(f"\n📝 Benchmark summary saved to: {output_txt}")
    return results


### --- 5️⃣ Run Benchmarking --- ###
if __name__ == "__main__":
    import time
    MOL_SIZE = 100000

    batch_size = 50
    num_samples_to_have = MOL_SIZE / batch_size
    num_samples = int(num_samples_to_have)
    real_amount = num_samples*batch_size
    if real_amount >= 100_000:
        batch_size = 10
    elif real_amount >= 10_000:
        batch_size = 20

    benchmark_dir = f"./benchmark_runs"

    # mol_sizes = [50,100,1_000,10_000,100_000,1_000_000]
    mol_sizes = [50,100,1_000,10_000,20_000,30_000,40_000,50_000,60_000,70_000,80_000,90_000,100_000]

    output_csv = f"{MOL_SIZE}_real_{real_amount}_generated_molecules.csv"
    # TODO: 06/22/2025 GET full dataset from singular csv

    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_PRETRAIN_model_small_adamw_earlyS/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_PRETRAIN_model_4_warmup/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_PRETRAIN_model_small_lamb_earlyS_extradropout_highLR_lowThresh/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_PRETRAIN_model_small_lamb_earlyS_extradropout_highLR_lowThresh_or7th/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # Finetuned
    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_finetune_model_baseline_tuned/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # Fine tuned best model 3x (increased 2x then 3x and yeah it was best)
    gen = HuggingFaceMoleculeGenerator(
        model_path="./runs/selfies_BART_finetune_model_small_adamw_earlyS_6_long_3x/model",
        tokenizer_path="full_tokenizer_finetune_and_pretrain",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


    start_time = time.time()


    # Loop through each molecule size and benchmark
    for mol_size in mol_sizes:
        num_samples = mol_size // batch_size
        real_amount = num_samples * batch_size

        run_dir = os.path.join(benchmark_dir, f"{mol_size}_mol_{batch_size}_batch")
        os.makedirs(run_dir, exist_ok=True)

        output_csv = os.path.join(run_dir, f"generated_{real_amount}_molecules.csv")
        output_txt = os.path.join(run_dir, "summary_statistics.txt")
        print(f"\n🚀 Running benchmark for {real_amount} molecules ({mol_size} target, batch size {batch_size})")
        start_time = time.time()


        # benchmark_results = benchmark_generated_molecules_selfies_parquet(
        #     gen,
        #     selfies_pt="./model_4_warmup_FP_pre_cleaned.parquet",
        #     num_samples=num_samples,
        #     batch_size=batch_size,
        #     output_csv=output_csv,
        #     output_txt=output_txt
        # )

        # benchmark_results = benchmark_generated_molecules_selfies_parquet_bloom(
        #     gen,
        #     selfies_pt="./model_4_warmup_FP_pre_cleaned.parquet",
        #     num_samples=num_samples,
        #     batch_size=batch_size,
        #     output_csv=output_csv,
        #     output_txt=output_txt
        # )

        benchmark_results = benchmark_generated_molecules_selfies_parquet_bloom(
            gen,
            selfies_pt="./model_4_warmup_FP_pre_cleaned.parquet",
            num_samples=num_samples,
            batch_size=batch_size,
            output_csv=output_csv,
            output_txt=output_txt,
            bloom_error_rate=0.05
        )



        # benchmark_results = benchmark_generated_molecules_selfies_parquet_cuckoo(
        #     gen,
        #     selfies_pt="./model_4_warmup_FP_pre_cleaned.parquet",
        #     num_samples=num_samples,
        #     batch_size=batch_size,
        #     output_csv=output_csv,
        #     output_txt=output_txt
        # )


        duration = round(time.time() - start_time, 2)
        print(f"✅ Completed benchmark for {mol_size} molecules in {duration} seconds.\nSaved to: {output_csv}")

        with open(output_txt, "a") as f:
            f.write(f"duration: {duration} s")

    print(f"\n⏳ Benchmark completed in {round(time.time() - start_time, 2)} seconds.")