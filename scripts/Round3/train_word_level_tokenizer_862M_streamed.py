import os
import selfies as sf
from tqdm import tqdm
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import dask.dataframe as dd

# ====== CONFIG ======
PARQUET_PATH = "./tokenizer_data/combined_selfies_dataset_for_tokenizer.parquet"
SELFIES_COLUMN = "selfies"
CORPUS_FILE = "full_tokenizer_finetune_and_pretrain.txt"
TOKENIZER_DIR = "full_tokenizer_finetune_and_pretrain"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"]
# =====================

# === STEP 1: Load Parquet file lazily and build corpus ===
print("🔄 Streaming Parquet partitions and generating tokenized corpus...")

df_dask = dd.read_parquet(PARQUET_PATH, columns=[SELFIES_COLUMN])
df_dask = df_dask.dropna(subset=[SELFIES_COLUMN])

vocab_counter = Counter()

# Open the file only once
with open(CORPUS_FILE, "w") as f:
    # Iterate over each Dask partition
    for delayed_partition in tqdm(df_dask.to_delayed()):
        df_partition = delayed_partition.compute()  # Only load 1 partition at a time

        for s in df_partition[SELFIES_COLUMN]:
            try:
                selfies_str = sf.encoder(sf.decoder(s))  # Normalize and validate
                tokens = list(sf.split_selfies(selfies_str))
                vocab_counter.update(tokens)
                f.write(" ".join(tokens) + "\n")
            except Exception:
                continue

# === STEP 2: Build vocab dictionary ===
print("\n📚 Building vocabulary...")
vocab = {token: idx + len(SPECIAL_TOKENS)
         for idx, (token, _) in enumerate(vocab_counter.most_common())}
for idx, token in enumerate(SPECIAL_TOKENS):
    vocab[token] = idx

# === STEP 3: Create WordLevel tokenizer ===
print("🔧 Creating tokenizer...")
model = WordLevel(vocab=vocab, unk_token="[UNK]")
tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = Whitespace()

# === STEP 4: Wrap with Hugging Face Tokenizer ===
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    bos_token="<s>",
    eos_token="</s>",
)

hf_tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"\n✅ Tokenizer saved to: {TOKENIZER_DIR}")

# === STEP 5: Test tokenizer ===
print("\n🔍 Testing tokenizer on example molecule...")

def selfies_encode(smiles):
    selfies = sf.encoder(smiles)
    tokens = list(sf.split_selfies(selfies))
    return selfies, tokens

example_smiles = "CCO"
selfies_str, tokens = selfies_encode(example_smiles)

print("SMILES:   ", example_smiles)
print("SELFIES:  ", selfies_str)
print("Tokens:   ", tokens)

# --- Method 1
ids_direct = hf_tokenizer.convert_tokens_to_ids(tokens)
print("\n[Method1] convert_tokens_to_ids →", ids_direct)

# --- Method 2
enc = tokenizer.encode(" ".join(tokens))
print("[Method2] tokenizer.encode.ids    →", enc.ids)
print("[Method2] tokenizer.encode.tokens →", enc.tokens)

# --- Method 3
batch = [tokens]
enc_batch = hf_tokenizer.encode_batch(batch, add_special_tokens=False)
print("[Method3] encode_batch.ids         →", enc_batch[0].ids)
print("[Method3] encode_batch.tokens      →", enc_batch[0].tokens)

# === STEP 6: Vocab check ===
print("\n🔎 Vocab sanity check:")
vocab_set = hf_tokenizer.get_vocab()
for t in tokens:
    print(f"{t}: {'FOUND' if t in vocab_set else 'MISSING'}")

