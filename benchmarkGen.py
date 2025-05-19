# import pandas as pd
# import torch
# import selfies as sf
# import multiprocessing as mp
# from rdkit import Chem
# from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
#
# def selfies_to_smiles(selfies_str):
#     """Convert a SELFIES string to a valid SMILES string."""
#     try:
#         smiles = sf.decoder(selfies_str)
#         if Chem.MolFromSmiles(smiles):  # Ensure it's valid
#             return smiles
#     except Exception:
#         return None
#     return None
#
# ### --- 1️⃣ Load SMILES Data from CSV --- ###
# def load_smiles_from_csv(csv_path, selfies_column="selfies"):
#     """
#     Load a set of SMILES from a CSV file that contains SELFIES.
#
#     Parameters:
#     - csv_path: Path to the CSV file
#     - selfies_column: Column name containing SELFIES strings
#
#     Returns:
#     - A set of unique SMILES
#     """
#     try:
#         df = pd.read_csv(csv_path, usecols=[selfies_column])  # Load only the SELFIES column
#         df["SMILES"] = df[selfies_column].astype(str).apply(selfies_to_smiles)  # Convert SELFIES to SMILES
#         smiles_set = set(df["SMILES"].dropna())  # Drop invalid conversions & convert to set
#         print(f"✅ Loaded {len(smiles_set):,} unique SMILES from {csv_path} (converted from SELFIES)")
#         return smiles_set
#     except Exception as e:
#         print(f"Error loading {csv_path}: {e}")
#         return set()
#
#
# ### --- 2️⃣ Define Model Wrapper --- ###
# class HuggingFaceMoleculeGenerator:
#     def __init__(self, model_path, tokenizer_path, device="cpu"):
#         """Load Hugging Face BART model & tokenizer for molecule generation."""
#         self.device = torch.device(device)
#         self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
#
#         # ✅ Define model configuration (must match training setup)
#         config = BartConfig(
#             vocab_size=360,
#             max_position_embeddings=514,
#             d_model=768,
#             encoder_layers=12,
#             decoder_layers=12,
#             encoder_attention_heads=12,
#             decoder_attention_heads=12,
#             encoder_ffn_dim=3072,
#             decoder_ffn_dim=3072,
#             activation_function="gelu",
#             pad_token_id=1,
#             eos_token_id=2,
#             bos_token_id=0,
#         )
#
#         # ✅ Load model and weights
#         self.model = BartForConditionalGeneration(config)
#         state_dict = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(state_dict, strict=False)
#         self.model.to(self.device)
#         self.model.eval()
#
#         print(f"✅ Fine-tuned model loaded from {model_path}")
#
#     def selfies_to_smiles(self, selfies_list):
#         """Convert SELFIES to valid SMILES."""
#         smiles_list = []
#         for selfies in selfies_list:
#             try:
#                 smiles = sf.decoder(selfies)
#                 if Chem.MolFromSmiles(smiles):  # Ensure valid molecule
#                     smiles_list.append(smiles)
#             except Exception:
#                 continue  # Skip invalid molecules
#         return smiles_list
#
#     def sample(self, n, batch_size=100):
#         """Generate `n` molecules in smaller batches and return SMILES strings."""
#         input_text = "[C][N]"  # Example SELFIES input
#         input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
#
#         generated_smiles = []
#
#         for _ in range(0, n, batch_size):
#             current_batch_size = min(batch_size, n - len(generated_smiles))  # Handle last batch
#
#             with torch.no_grad():
#                 output_ids = self.model.generate(
#                     input_ids.expand(current_batch_size, -1),  # Duplicate input for batch processing
#                     max_length=500,
#                     num_return_sequences=1,  # Now handled by batch_size
#                     do_sample=True,
#                     temperature=2.5,
#                     top_k=50,
#                     top_p=0.95,
#                     repetition_penalty=2.3,
#                     num_beams=1
#                 )
#
#             selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#             smiles_list = self.selfies_to_smiles(selfies_list)
#
#             generated_smiles.extend([s for s in smiles_list if s is not None])  # Filter out invalid molecules
#
#         return generated_smiles
#
#
# ### --- 3️⃣ Benchmarking: Compute Uniqueness & Novelty --- ###
# def is_valid_smiles(smiles):
#     """Check if a SMILES string is valid using RDKit."""
#     return Chem.MolFromSmiles(smiles) is not None
#
#
# def check_novelty(smiles_list, train_set, pretrain_set):
#     """Parallelized function to check novelty of molecules."""
#     return [smiles for smiles in smiles_list if smiles not in train_set and smiles not in pretrain_set]
#
#
# def benchmark_generated_molecules(gen, train_set, pretrain_set, num_samples=5000, batch_size=100,
#                                   num_workers=mp.cpu_count()):
#     """
#     Benchmark generated molecules for % uniqueness and % novelty with multiprocessing.
#
#     Parameters:
#     - gen: HuggingFaceMoleculeGenerator instance
#     - train_set: Set of SMILES from fine-tuned training dataset
#     - pretrain_set: Set of SMILES from pretraining dataset
#     - num_samples: Total number of molecules to generate
#     - batch_size: Number of molecules generated per batch
#     - num_workers: Number of CPU cores for multiprocessing
#
#     Returns:
#     - A dictionary with uniqueness and novelty scores
#     """
#
#     # ✅ Generate molecules in batches
#     generated_molecules = gen.sample(num_samples, batch_size=batch_size)
#
#     # ✅ Step 1: Filter valid molecules using multiprocessing
#     print("🔍 Filtering valid molecules...")
#     with mp.Pool(num_workers) as pool:
#         valid_molecules = list(filter(None, pool.map(is_valid_smiles, generated_molecules)))
#
#     print(f"✅ Valid molecules: {len(valid_molecules)} / {len(generated_molecules)}")
#
#     # ✅ Step 2: Compute uniqueness (% of valid molecules that are unique)
#     unique_molecules = list(set(valid_molecules))  # Remove duplicates
#     uniqueness = (len(unique_molecules) / len(valid_molecules)) * 100 if valid_molecules else 0
#
#     # ✅ Step 3: Compute novelty (% of unique molecules not in training/pretraining datasets)
#     print("🔍 Checking novelty...")
#     with mp.Pool(num_workers) as pool:
#         novel_molecules = pool.starmap(check_novelty, [(unique_molecules, train_set, pretrain_set)])
#
#     novel_molecules = novel_molecules[0]  # Extract from list of lists
#     novelty = (len(novel_molecules) / len(unique_molecules)) * 100 if unique_molecules else 0
#
#     # ✅ Print and return results
#     results = {
#         "Total Generated": len(generated_molecules),
#         "Valid Unique Molecules": len(unique_molecules),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel Molecules": len(novel_molecules),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     for key, value in results.items():
#         print(f"{key}: {value}")
#
#     return results
#
#
# ### --- 4️⃣ Run Benchmarking --- ###
# if __name__ == "__main__":
#     import time
#
#     # Load pretraining and fine-tuning datasets
#     pretrain_smiles = load_smiles_from_csv("data/trainable_selfies_model_nada.csv")
#     finetune_smiles = load_smiles_from_csv("data/smiles_finetune_data_properties_selfies.csv")
#
#     # Initialize model generator
#     gen = HuggingFaceMoleculeGenerator(
#         model_path="./selfies_BART_finetuned_model_nada.pth",
#         tokenizer_path="./data/bpe_filter_model_nada",
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
#
#     # Run benchmark
#     start_time = time.time()
#     benchmark_results = benchmark_generated_molecules(
#         gen=gen,
#         train_set=finetune_smiles,
#         pretrain_set=pretrain_smiles,
#         num_samples=100000,
#         batch_size=100,
#         num_workers=24  # Adjust for your system
#     )
#     end_time = time.time()
#
#     print(f"\n⏳ Benchmark completed in {round(end_time - start_time, 2)} seconds.")

import pandas as pd
import torch
import selfies as sf
import multiprocessing as mp
from rdkit import Chem
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers.models.auto.tokenization_auto import PreTrainedTokenizerFast


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
    """Standardize SMILES format for uniqueness checking."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


### --- 2️⃣ Load SMILES Data from CSV (Converted from SELFIES) --- ###
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

        # # ✅ Define model configuration (must match training setup)
        # config = BartConfig(
        #     vocab_size=360,
        #     max_position_embeddings=514,
        #     d_model=768,
        #     encoder_layers=12,
        #     decoder_layers=12,
        #     encoder_attention_heads=12,
        #     decoder_attention_heads=12,
        #     encoder_ffn_dim=3072,
        #     decoder_ffn_dim=3072,
        #     activation_function="gelu",
        #     pad_token_id=1,
        #     eos_token_id=2,
        #     bos_token_id=0,
        # )

        # ✅ Load model and weights
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        # state_dict = torch.load(model_path, map_location=self.device)
        # self.model.load_state_dict(state_dict, strict=False)
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

    def sample(self, n, batch_size=100):
        """Generate `n` molecules in smaller batches and return SMILES strings."""
        input_text = "[C][N]"  # Example SELFIES input # DOESN'T DO A DIRECT STARTING BECAUSE OF THE BART MODEL STRUCTURE... 03-07-2025 TODO: ELABORATE ON THIS!
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        generated_smiles = []

        for _ in range(0, n, batch_size):
            current_batch_size = min(batch_size, n - len(generated_smiles))  # Handle last batch

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids.expand(current_batch_size, -1),  # Duplicate input for batch processing
                    max_length=500,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=2.5,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=2.3,
                    num_beams=1
                )

            selfies_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            smiles_list = self.selfies_to_smiles(selfies_list)

            generated_smiles.extend([s for s in smiles_list if s is not None])  # Filter out invalid molecules

        return generated_smiles

### --- 4️⃣ Benchmarking: Compute Uniqueness & Novelty and Save to CSV --- ###
def is_valid_smiles(smiles):
    """Check if a SMILES string is valid using RDKit."""
    return Chem.MolFromSmiles(smiles) is not None

def check_novelty(smiles_list, train_set, pretrain_set):
    """Parallelized function to check novelty of molecules."""
    return [smiles for smiles in smiles_list if smiles not in train_set and smiles not in pretrain_set]

# def benchmark_generated_molecules(gen, train_set, pretrain_set, num_samples=5000, batch_size=100,
#                                   num_workers=mp.cpu_count(), output_csv="generated_molecules.csv"):
#     """
#     Benchmark generated molecules for % uniqueness and % novelty with multiprocessing.
#
#     Parameters:
#     - gen: HuggingFaceMoleculeGenerator instance
#     - train_set: Set of SMILES from fine-tuned training dataset
#     - pretrain_set: Set of SMILES from pretraining dataset
#     - num_samples: Total number of molecules to generate
#     - batch_size: Number of molecules generated per batch
#     - num_workers: Number of CPU cores for multiprocessing
#     - output_csv: Path to save generated molecules
#
#     Returns:
#     - A dictionary with uniqueness and novelty scores
#     """
#
#     # ✅ Generate molecules in batches
#     generated_molecules = gen.sample(num_samples, batch_size=batch_size)
#
#     # ✅ Step 1: Filter valid molecules using multiprocessing
#     print("🔍 Filtering valid molecules...")
#     with mp.Pool(num_workers) as pool:
#         valid_molecules = list(filter(None, pool.map(is_valid_smiles, generated_molecules)))
#
#     print(f"✅ Valid molecules: {len(valid_molecules)} / {len(generated_molecules)}")
#
#     # ✅ Step 2: Compute uniqueness
#     unique_molecules = list(set(valid_molecules))  # Remove duplicates
#     uniqueness = (len(unique_molecules) / len(valid_molecules)) * 100 if valid_molecules else 0
#
#     # ✅ Step 3: Compute novelty
#     print("🔍 Checking novelty...")
#     with mp.Pool(num_workers) as pool:
#         novel_molecules = pool.starmap(check_novelty, [(unique_molecules, train_set, pretrain_set)])
#
#     novel_molecules = novel_molecules[0]  # Extract from list of lists
#     novelty = (len(novel_molecules) / len(unique_molecules)) * 100 if unique_molecules else 0
#
#     # ✅ Save results to CSV
#     df = pd.DataFrame({
#         "SMILES": unique_molecules,
#         "Novel": [smiles in novel_molecules for smiles in unique_molecules],
#         "Unique": [smiles in unique_molecules for smiles in unique_molecules]
#     })
#     df.to_csv(output_csv, index=False)
#     print(f"📁 Generated molecules saved to: {output_csv}")
#
#     # ✅ Print and return results
#     results = {
#         "Total Generated": len(generated_molecules),
#         "Valid Unique Molecules": len(unique_molecules),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel Molecules": len(novel_molecules),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     for key, value in results.items():
#         print(f"{key}: {value}")
#
#     return results

# def benchmark_generated_molecules(gen, train_set, pretrain_set, num_samples=5000, batch_size=100,
#                                   num_workers=mp.cpu_count(), output_csv="generated_molecules.csv"):
#     """
#     Benchmark generated molecules for % uniqueness and % novelty with multiprocessing.
#     """
#     # ✅ Generate molecules in batches
#     generated_molecules = gen.sample(num_samples, batch_size=batch_size)
#
#     # 🔍 DEBUG: Print first 10 generated molecules
#     print("\n🔍 First 10 Generated Molecules:")
#     print(generated_molecules[:10])
#
#     # ✅ Step 1: Filter valid molecules using multiprocessing
#     print("🔍 Filtering valid molecules...")
#     with mp.Pool(num_workers) as pool:
#         valid_molecules = list(filter(None, pool.map(is_valid_smiles, generated_molecules)))
#
#     print(f"✅ Valid molecules: {len(valid_molecules)} / {len(generated_molecules)}")
#
#     # ✅ Step 2: Convert to canonical SMILES for uniqueness checking
#     print("🔍 Standardizing SMILES...")
#     with mp.Pool(num_workers) as pool:
#         canonical_smiles = list(filter(None, pool.map(canonicalize_smiles, valid_molecules)))
#
#     unique_molecules = list(set(canonical_smiles))  # Remove duplicates
#     uniqueness = (len(unique_molecules) / len(valid_molecules)) * 100 if valid_molecules else 0
#
#     # ✅ Step 3: Compute novelty efficiently
#     print("🔍 Checking novelty...")
#     novel_molecules = [mol for mol in unique_molecules if mol not in train_set and mol not in pretrain_set]
#     novelty = (len(novel_molecules) / len(unique_molecules)) * 100 if unique_molecules else 0
#
#     # ✅ Save results to CSV
#     df = pd.DataFrame({
#         "SMILES": unique_molecules,
#         "Novel": [mol in novel_molecules for mol in unique_molecules],
#         "Unique": [mol in unique_molecules for mol in unique_molecules]
#     })
#     df.to_csv(output_csv, index=False)
#     print(f"📁 Generated molecules saved to: {output_csv}")
#
#     # ✅ Print and return results
#     results = {
#         "Total Generated": len(generated_molecules),
#         "Valid Unique Molecules": len(unique_molecules),
#         "% Uniqueness": round(uniqueness, 3),
#         "Novel Molecules": len(novel_molecules),
#         "% Novelty": round(novelty, 3),
#     }
#
#     print("\n📊 Benchmark Results:")
#     for key, value in results.items():
#         print(f"{key}: {value}")
#
#     return results

def benchmark_generated_molecules(gen, train_set, pretrain_set, num_samples=5000, batch_size=100,
                                  num_workers=mp.cpu_count(), output_csv="generated_molecules.csv"):
    """
    Benchmark generated molecules for % uniqueness and % novelty with multiprocessing.
    """
    # ✅ Generate molecules in batches
    generated_molecules = gen.sample(num_samples, batch_size=batch_size)

    # ✅ Step 1: Remove empty & invalid molecules
    filtered_molecules = [mol for mol in generated_molecules if mol and is_valid_smiles(mol)]
    print(f"✅ Valid molecules after removing empty: {len(filtered_molecules)} / {len(generated_molecules)}")

    # ✅ Step 2: Convert to canonical SMILES for uniqueness checking
    print("🔍 Standardizing SMILES...")
    with mp.Pool(num_workers) as pool:
        canonical_smiles = list(filter(None, pool.map(canonicalize_smiles, filtered_molecules)))

    unique_smiles_set = set(canonical_smiles)  # Remove duplicates using set()
    unique_molecules = list(unique_smiles_set)
    uniqueness = (len(unique_molecules) / len(filtered_molecules)) * 100 if filtered_molecules else 0

    # ✅ Step 3: Compute novelty efficiently
    print("🔍 Checking novelty...")
    train_and_pretrain = train_set | pretrain_set  # Merge sets for faster lookup
    novel_molecules = list(unique_smiles_set - train_and_pretrain)  # Faster novelty check
    novelty = (len(novel_molecules) / len(unique_molecules)) * 100 if unique_molecules else 0

    # ✅ Save results to CSV
    df = pd.DataFrame({
        "SMILES": unique_molecules,
        "Novel": [mol in novel_molecules for mol in unique_molecules],
        "Unique": [mol in unique_smiles_set for mol in unique_molecules]
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Generated molecules saved to: {output_csv}")

    # ✅ Print and return results
    results = {
        "Total Generated": len(generated_molecules),
        "Valid Unique Molecules": len(unique_molecules),
        "% Uniqueness": round(uniqueness, 3),
        "Novel Molecules": len(novel_molecules),
        "% Novelty": round(novelty, 3),
    }

    print("\n📊 Benchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


### --- 5️⃣ Run Benchmarking --- ###
if __name__ == "__main__":
    import time

    pretrain_smiles = load_smiles_from_csv("data/trainable_selfies_model_nada.csv")
    finetune_smiles = load_smiles_from_csv("data/smiles_finetune_data_properties_selfies.csv")

    gen = HuggingFaceMoleculeGenerator(
        model_path="./DataForGen/selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw",
        tokenizer_path="./DataForGen/selfies_word_tokenizer",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    benchmark_results = benchmark_generated_molecules(
        gen, finetune_smiles, pretrain_smiles, num_samples=1000, batch_size=100, num_workers=12,
        output_csv="generated_molecules_1k_new.csv"
    )

    print(f"\n⏳ Benchmark completed in {round(time.time() - start_time, 2)} seconds.")
