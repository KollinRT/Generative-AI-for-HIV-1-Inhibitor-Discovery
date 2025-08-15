import os
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    TypedDict,
)

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    LogitsProcessorList,
)

from token_mapping import token_mapping

import dask.dataframe as dd

import gc

from pybloom_live import ScalableBloomFilter


from utils import MaskTokensLogitsProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# For Generative Results
class BenchmarkResults(TypedDict):
    Total_Generated: int
    Valid_Unique_SELFIES: int
    Percent_Uniqueness: float
    Novel_Molecules: int
    Percent_Novelty: float


# def build_bloom_filter_from_parquet(parquet_path, column="selfies", error_rate=0.001):
def build_bloom_filter_from_parquet(
    parquet_path: str, column: str = "selfies", error_rate: float = 0.001
) -> ScalableBloomFilter:
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


def apply_selfies_token_mapping(selfies_str: str) -> str:
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
def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Convert a SELFIES string to a valid SMILES string."""
    try:
        smiles = sf.decoder(selfies_str)
        if Chem.MolFromSmiles(smiles):  # Ensure it's valid
            return smiles
    except Exception:
        return None
    return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
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
def load_smiles_from_csv(csv_path: str, selfies_column: str = "selfies") -> Set[str]:
    """
    Load a set of SMILES from a CSV file that contains SELFIES.

    Parameters:
    - csv_path: Path to the CSV file
    - selfies_column: Column name containing SELFIES strings

    Returns:
    - A set of unique SMILES
    """
    try:
        df = pd.read_csv(
            csv_path, usecols=[selfies_column]
        )  # Load only the SELFIES column
        df["SMILES"] = (
            df[selfies_column].astype(str).apply(selfies_to_smiles)
        )  # Convert SELFIES to SMILES
        # smiles_set = set(
        #     df["SMILES"].dropna()
        # )  # Drop invalid conversions & convert to set
        smiles_set: Set[str] = set(df["SMILES"].dropna())  # type: ignore[arg-type]
        print(
            f"✅ Loaded {len(smiles_set):,} unique SMILES from {csv_path} (converted from SELFIES)"
        )
        return smiles_set
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return set()


### --- 3️⃣ Define Model Wrapper for Generation --- ###
class HuggingFaceMoleculeGenerator:
    # def __init__(self, model_path, tokenizer_path, device="cuda"):
    #     """Load Hugging Face BART model & tokenizer for molecule generation."""
    #     self.device = torch.device(device)
    #     self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    #
    #     # ✅ Load model and weights
    #     self.model = BartForConditionalGeneration.from_pretrained(model_path)
    #     self.model.to(self.device)
    #     self.model.eval()
    #
    #     print(f"✅ Fine-tuned model loaded from {model_path}")
    # def __init__(
    #     self,
    #     model_path,
    #     tokenizer_path,
    #     device="cuda",
    #     forbidden_tokens=None,
    #     forbidden_token_ids=None,
    # ):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        forbidden_tokens: Optional[Sequence[str]] = None,
        forbidden_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        # self.device = torch.device(device)
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        # self.model = (
        #     BartForConditionalGeneration.from_pretrained(model_path)
        #     .to(self.device)
        #     .eval()
        # )
        self.device: torch.device = torch.device(device)
        self.tokenizer: PreTrainedTokenizerFast = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        )
        self.model: BartForConditionalGeneration = (
            BartForConditionalGeneration.from_pretrained(model_path)
            .to(self.device)
            .eval()
        )

        # Configure forbidden ids (either strings or ids)
        # ids_from_strings = []
        # if forbidden_tokens:
        #     # Each string can be one token or multiple; we only suppress single-token strings here.
        #     # For multi-token phrases, see option B below.
        #     ids_from_strings = self.tokenizer.convert_tokens_to_ids(forbidden_tokens)
        #     ids_from_strings = [
        #         i
        #         for i in ids_from_strings
        #         if i is not None and i != self.tokenizer.unk_token_id
        #     ]
        #
        # self.forbidden_token_ids = set((forbidden_token_ids or []) + ids_from_strings)
        # Configure forbidden ids (either strings or ids)
        ids_from_strings: List[Optional[int]] = []
        if forbidden_tokens:
            # Each string can be one token or multiple; we only suppress single-token strings here.
            # ids_from_strings = self.tokenizer.convert_tokens_to_ids(list(forbidden_tokens))
            ids_from_strings = self.tokenizer.convert_tokens_to_ids(forbidden_tokens)
        cleaned_from_strings: List[int] = [
            i
            for i in ids_from_strings
            if i is not None and i != self.tokenizer.unk_token_id
        ]

        self.forbidden_token_ids: Set[int] = set((forbidden_token_ids or [])) | set(
            cleaned_from_strings
        )

        print(f"✅ Fine-tuned model loaded from {model_path}")
        if self.forbidden_token_ids:
            print(
                f"🚫 Will suppress {len(self.forbidden_token_ids)} token ids during generation."
            )

    def selfies_to_smiles(self, selfies_list: Sequence[str]) -> List[str]:
        """Convert SELFIES to valid SMILES."""
        # smiles_list = []
        smiles_list: List[str] = []
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
    #                 max_length=63,
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
    #         # mapped_selfies_list = [apply_selfies_token_mapping(s) for s in selfies_list]
    #
    #         # ✅ Convert SELFIES to valid SMILES
    #         # smiles_list = self.selfies_to_smiles(mapped_selfies_list)
    #         # all_smiles.extend(smiles_list)
    #
    #
    #         # all_selfies_list = []
    #         # for molecules in all_selfies:
    #         #     all_selfies.append(molecules.replace(" ", ""))
    #
    #         all_selfies_list = []
    #         for molecules in selfies_list:
    #             all_selfies_list.append(molecules.replace(" ", ""))
    #
    #         # all_selfies.extend(selfies_list)
    #         all_selfies.extend(all_selfies_list)
    #
    #         # TODO: 08/01/25 16:45 pm: ALSO CONSIDER DOING MAPPED SELFIES HERE!
    #
    #         # # ✅ Save to file if requested
    #         # if output_csv:
    #         #     with open(output_csv, "a") as f:
    #         #         for s in selfies_list:
    #         #             f.write(f"{s},True,Novel\n")
    #
    #         # ✅ Save to file if requested
    #         if output_csv:
    #             with open(output_csv, "a") as f:
    #                 for s in all_selfies:
    #                     f.write(f"{s},True,Novel\n")
    #
    #         total_generated += current_batch_size
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #
    #     return all_selfies
    #
    #     # return all_smiles
    #

    # def sample(self, n, batch_size=100, prefix="", output_csv="output2.csv"):
    def sample(
        self,
        n: int,
        batch_size: int = 100,
        prefix: str = "",
        output_csv: Optional[str] = "output2.csv",
    ) -> List[str]:
        if not prefix.strip():
            print("⚠️ Empty prefix provided. Using default '<s>'.")
            prefix = "<s>"

        tokens = prefix.strip().split()
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Build logits processors
        processors = LogitsProcessorList()
        # (optional) don’t stop too early
        # processors.append(MinLengthLogitsProcessor(min_length=4, eos_token_id=self.tokenizer.eos_token_id))
        # hard-mask forbidden tokens
        if self.forbidden_token_ids:
            processors.append(MaskTokensLogitsProcessor(self.forbidden_token_ids))

        total_generated = 0
        # all_selfies = []
        all_selfies: List[str] = []

        while total_generated < n:
            current_batch_size = min(batch_size, n - total_generated)
            with torch.no_grad():
                # output_ids = self.model.generate(
                #     input_ids=input_tensor.expand(current_batch_size, -1),
                #     max_length=63,
                #     num_return_sequences=current_batch_size,
                #     do_sample=True,
                #     temperature=1.5,
                #     top_k=30,
                #     top_p=0.9,
                #     repetition_penalty=1.1,
                #     logits_processor=processors,   # ← key line
                #     eos_token_id=self.tokenizer.eos_token_id,
                #     pad_token_id=self.tokenizer.pad_token_id
                # )
                output_ids = self.model.generate(
                    input_ids=input_tensor.expand(current_batch_size, -1),
                    max_length=63,
                    num_return_sequences=current_batch_size,
                    do_sample=True,
                    temperature=1.5,
                    top_k=30,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    logits_processor=processors,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # new: ensure at least 4 tokens are generated past the prompt
                    min_new_tokens=4,
                )

            selfies_list = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            all_selfies.extend(s.replace(" ", "") for s in selfies_list)

            if output_csv:
                with open(output_csv, "a") as f:
                    for s in selfies_list:
                        f.write(f"{s},True,Novel\n")

            total_generated += current_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        return all_selfies


### --- 4️⃣ Benchmarking: Compute Uniqueness & Novelty and Save to CSV --- ###
def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid using RDKit.
    Args:
        smiles:

    Returns:
        Any:
    """
    return Chem.MolFromSmiles(smiles) is not None


# def check_novelty(smiles_list, train_set, pretrain_set):
def check_novelty(
    smiles_list: Sequence[str], train_set: Set[str], pretrain_set: Set[str]
) -> List[str]:
    """Parallelized function to check novelty of molecules.

    Args:
        smiles_list:
        train_set:
        pretrain_set:

    Returns:
        list[Any]:
    """
    return [
        smiles
        for smiles in smiles_list
        if smiles not in train_set and smiles not in pretrain_set
    ]


def benchmark_generated_molecules_selfies_parquet(
    gen: HuggingFaceMoleculeGenerator,
    selfies_pt: Union[str, pd.DataFrame],
    num_samples: int = 5000,
    batch_size: int = 100,
    output_csv: str = "generated_molecules.csv",
    output_txt: str = "generated_molecules.txt",
    temp_train_set_path: str = "train_selfies.txt",
) -> BenchmarkResults:
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
            temp_train_set_path, single_file=True, index=False, header=False
        )

    # Step 2: Load training SELFIES from disk (now small)
    with open(temp_train_set_path, "r") as f:
        # train_selfies_set = set(line.strip() for line in f if line.strip())
        train_selfies_set: Set[str] = set(line.strip() for line in f if line.strip())
    print(f"✅ Loaded {len(train_selfies_set)} unique training SELFIES")

    # Step 3: Generate molecules (SELFIES)
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")

    # Step 4: Uniqueness
    # unique_selfies_set = set(generated_selfies)
    # uniqueness = (
    #     (len(unique_selfies_set) / len(generated_selfies)) * 100
    #     if generated_selfies
    #     else 0
    # )
    unique_selfies_set: Set[str] = set(generated_selfies)
    uniqueness: float = (
        (len(unique_selfies_set) / len(generated_selfies) * 100)
        if generated_selfies
        else 0.0
    )

    # Step 5: Novelty
    # novel_selfies = list(unique_selfies_set - train_selfies_set)
    # novelty = (
    #     (len(novel_selfies) / len(unique_selfies_set)) * 100
    #     if unique_selfies_set
    #     else 0
    # )
    novel_selfies: List[str] = list(unique_selfies_set - train_selfies_set)
    novelty: float = (
        (len(novel_selfies) / len(unique_selfies_set) * 100)
        if unique_selfies_set
        else 0.0
    )

    # Step 6: Save output CSV
    df = pd.DataFrame(
        {
            "SELFIES": list(unique_selfies_set),
            "Novel": [s in novel_selfies for s in unique_selfies_set],
            "Unique": [True] * len(unique_selfies_set),
        }
    )
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
    bloom_error_rate=0.001,
):
    """
    Benchmarks generated SELFIES using a Bloom filter for novelty checking.
    """

    # ✅ Step 1: Load training data into Bloom filter
    if isinstance(selfies_pt, str):
        if not os.path.exists(selfies_pt):
            raise FileNotFoundError(f"Parquet file not found: {selfies_pt}")
        bloom_filter = build_bloom_filter_from_parquet(
            selfies_pt, error_rate=bloom_error_rate
        )
    elif isinstance(selfies_pt, pd.DataFrame):
        selfies_series = selfies_pt["selfies"].dropna().astype(str)
        bloom_filter = ScalableBloomFilter(
            initial_capacity=1_000_000_000, error_rate=bloom_error_rate
        )
        for s in selfies_series:
            bloom_filter.add(s)
    else:
        raise ValueError(
            "`selfies_pt` must be a path to a Parquet file or a pandas DataFrame."
        )

    # ✅ Step 2: Generate SELFIES strings
    generated_selfies = gen.sample(num_samples, batch_size=batch_size, prefix="")

    # ✅ Step 3: Uniqueness
    unique_selfies_set = set(generated_selfies)
    uniqueness = (
        (len(unique_selfies_set) / len(generated_selfies)) * 100
        if generated_selfies
        else 0
    )

    # ✅ Step 4: Novelty using Bloom filter
    novel_selfies = [s for s in unique_selfies_set if s not in bloom_filter]
    novelty = (
        (len(novel_selfies) / len(unique_selfies_set)) * 100
        if unique_selfies_set
        else 0
    )

    # ✅ Step 5: Save to CSV
    df = pd.DataFrame(
        {
            "SELFIES": list(unique_selfies_set),
            "Novel": [s in novel_selfies for s in unique_selfies_set],
            "Unique": [True] * len(unique_selfies_set),
        }
    )
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

    # MOL_SIZE = 100000
    MOL_SIZE: int = 100_000

    # batch_size = 50
    # num_samples_to_have = MOL_SIZE / batch_size
    # num_samples = int(num_samples_to_have)
    # real_amount = num_samples * batch_size
    # if real_amount >= 100_000:
    #     batch_size = 10
    # elif real_amount >= 10_000:
    #     batch_size = 20
    batch_size: int = 50
    num_samples_to_have: float = MOL_SIZE / batch_size
    num_samples: int = int(num_samples_to_have)
    real_amount: int = num_samples * batch_size
    if real_amount >= 100_000:
        batch_size = 10
    elif real_amount >= 10_000:
        batch_size = 20

    benchmark_dir = "./benchmark_runs"

    # mol_sizes = [50,100,1_000,10_000,100_000,1_000_000]
    # mol_sizes = [
    #     50,
    #     100,
    #     1_000,
    #     10_000,
    #     20_000,
    #     30_000,
    #     40_000,
    #     50_000,
    #     60_000,
    #     70_000,
    #     80_000,
    #     90_000,
    #     100_000,
    # ]
    mol_sizes: List[int] = [
        50,
        100,
        1_000,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
    ]

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
    # gen = HuggingFaceMoleculeGenerator(
    #     model_path="./runs/selfies_BART_finetune_model_small_adamw_earlyS_6_long_3x/model",
    #     tokenizer_path="full_tokenizer_finetune_and_pretrain",
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # forbidden = [
    #     "[Branch1_1]", "[Branch1_2]",
    #     # ring closures, charged atoms, or any artifacts you don’t want in benchmarks:
    #     "[Ring1]", "[Ring2]", "[+]", "[-]", "[O-]", "[N+]",
    #     # punctuation/sentinels that should never appear:
    #     ".", "<unk>"
    # ]

    # forbidden = [
    #     # Some ions not present in finetuning dataset
    #     "[Ag-4]",
    #     "[Rb+1]",
    #     "[Sn+3]",
    #     # Removed additional "." character
    #     ".",
    # ]
    forbidden: List[str] = [
        # Some ions not present in finetuning dataset
        "[Ag-4]",
        "[Rb+1]",
        "[Sn+3]",
        # Removed additional "." character
        ".",
    ]

    gen = HuggingFaceMoleculeGenerator(
        model_path="./runs/selfies_BART_finetune_model_small_adamw_earlyS_6_long_3x/model",
        tokenizer_path="full_tokenizer_finetune_and_pretrain",
        device="cuda" if torch.cuda.is_available() else "cpu",
        forbidden_tokens=forbidden,
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
        print(
            f"\n🚀 Running benchmark for {real_amount} molecules ({mol_size} target, batch size {batch_size})"
        )
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
            bloom_error_rate=0.05,
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
        print(
            f"✅ Completed benchmark for {mol_size} molecules in {duration} seconds.\nSaved to: {output_csv}"
        )

        with open(output_txt, "a") as f:
            f.write(f"duration: {duration} s")

    print(f"\n⏳ Benchmark completed in {round(time.time() - start_time, 2)} seconds.")
