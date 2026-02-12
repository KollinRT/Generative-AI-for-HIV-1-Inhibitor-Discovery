import os
import pandas as pd
import selfies as sf
from tqdm import tqdm
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# ====== CONFIG ======
CSV_PATH = "combined_selfies_dataset.csv"  # Path to your CSV file
SELFIES_COLUMN = "selfies"  # Column name containing SELFIES strings
CORPUS_FILE = "selfies_token_corpus_12M.txt"  # Intermediate corpus file
TOKENIZER_DIR = "selfies_word_tokenizer_12M"  # Output directory
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"]
# =====================

# === STEP 1: Build corpus and vocab ===
print("Generating tokenized corpus from CSV...")
df = pd.read_csv(CSV_PATH, usecols=[SELFIES_COLUMN])
df = df.dropna()

vocab_counter = Counter()
with open(CORPUS_FILE, "w") as f:
    for s in tqdm(df[SELFIES_COLUMN]):
        try:
            selfies_str = sf.encoder(sf.decoder(s))  # Normalize and validate
            tokens = list(sf.split_selfies(selfies_str))
            vocab_counter.update(tokens)
            f.write(" ".join(tokens) + "\n")
        except Exception:
            continue

# === STEP 2: Build vocab dictionary ===
print("\n Building vocabulary...")
vocab = {
    token: idx + len(SPECIAL_TOKENS)
    for idx, (token, _) in enumerate(vocab_counter.most_common())
}
for idx, token in enumerate(SPECIAL_TOKENS):
    vocab[token] = idx

# === STEP 3: Create WordLevel tokenizer with explicit vocab ===
print("Creating tokenizer...")
model = WordLevel(vocab=vocab, unk_token="[UNK]")
tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = Whitespace()

# === STEP 4: Hugging Face wrapper ===
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
print(f"\n Tokenizer saved to: {TOKENIZER_DIR}")

# === STEP 5: Test tokenizer on example molecule ===
print("\n Testing tokenizer on example molecule...")


def selfies_encode(smiles):
    selfies = sf.encoder(smiles)
    tokens = list(sf.split_selfies(selfies))
    return selfies, tokens


example_smiles = "CCO"
selfies_str, tokens = selfies_encode(example_smiles)

print("SMILES:   ", example_smiles)
print("SELFIES:  ", selfies_str)
print("Tokens:   ", tokens)

# --- METHOD 1: Direct HF lookup
ids_direct = hf_tokenizer.convert_tokens_to_ids(tokens)
print("\n[Method1] convert_tokens_to_ids →", ids_direct)

# --- METHOD 2: Underlying Tokenizer.encode
enc = tokenizer.encode(" ".join(tokens))
print("[Method2] tokenizer.encode.ids    →", enc.ids)
print("[Method2] tokenizer.encode.tokens →", enc.tokens)

# --- METHOD 3: HF wrapper encode_batch
batch = [tokens]  # one sequence, already split
enc_batch = hf_tokenizer.encode_batch(batch, add_special_tokens=False)
print("[Method3] encode_batch.ids         →", enc_batch[0].ids)
print("[Method3] encode_batch.tokens      →", enc_batch[0].tokens)

# === STEP 6: Vocab sanity check ===
print("\n Vocab sanity check:")
vocab_set = hf_tokenizer.get_vocab()
for t in tokens:
    print(f"{t}: {'FOUND' if t in vocab_set else 'MISSING'}")
