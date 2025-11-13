import os
import selfies as sf
from tqdm import tqdm
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import dask.dataframe as dd
from dask import delayed, compute

# ====== CONFIG ======
PARQUET_PATH = "./tokenizer_data/combined_selfies_dataset_for_tokenizer.parquet"
SELFIES_COLUMN = "selfies"
CORPUS_FILE = "full_tokenizer_finetune_and_pretrain.txt"
TOKENIZER_DIR = "full_tokenizer_finetune_and_pretrain"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"]
# =====================

# === STEP 1: Load Parquet file lazily ===
print("Streaming Parquet partitions and generating tokenized corpus...")

df_dask = dd.read_parquet(PARQUET_PATH, columns=[SELFIES_COLUMN])
df_dask = df_dask.dropna(subset=[SELFIES_COLUMN])


# === STEP 2: Define partition processing (NO encoding/decoding) ===
def process_partition(df_partition):
    local_counter = Counter()
    lines = []
    for selfies_str in df_partition[SELFIES_COLUMN]:
        try:
            tokens = list(sf.split_selfies(selfies_str))
            local_counter.update(tokens)
            lines.append(" ".join(tokens))
        except Exception:
            continue
    return local_counter, lines


# === STEP 3: Schedule partitions for parallel processing ===
tasks = []
for delayed_partition in tqdm(df_dask.to_delayed(), desc="🔧 Scheduling partitions"):
    task = delayed(lambda part: process_partition(part))(delayed_partition)
    tasks.append(task)

# === STEP 4: Run in parallel ===
print("Running tokenization in parallel...")
results = compute(*tasks, scheduler="threads")  # Or "processes"

# === STEP 5: Merge counters + write corpus ===
vocab_counter = Counter()
with open(CORPUS_FILE, "w") as f:
    for part_counter, lines in tqdm(results, desc="📝 Writing corpus"):
        vocab_counter.update(part_counter)
        for line in lines:
            f.write(line + "\n")

# === STEP 6: Build vocab dictionary ===
print("\n Building vocabulary...")
vocab = {
    token: idx + len(SPECIAL_TOKENS)
    for idx, (token, _) in enumerate(vocab_counter.most_common())
}
for idx, token in enumerate(SPECIAL_TOKENS):
    vocab[token] = idx

# === STEP 7: Create WordLevel tokenizer ===
print("Creating tokenizer...")
model = WordLevel(vocab=vocab, unk_token="[UNK]")
tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = Whitespace()

# === STEP 8: Hugging Face wrapper ===
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

# === STEP 9: Test tokenizer ===
print("\n Testing tokenizer on example SELFIES string...")

example_selfies = sf.encoder("CCO")
tokens = list(sf.split_selfies(example_selfies))

print("SELFIES:  ", example_selfies)
print("Tokens:   ", tokens)

ids_direct = hf_tokenizer.convert_tokens_to_ids(tokens)
print("\n[Method1] convert_tokens_to_ids →", ids_direct)

enc = tokenizer.encode(" ".join(tokens))
print("[Method2] tokenizer.encode.ids    →", enc.ids)
print("[Method2] tokenizer.encode.tokens →", enc.tokens)

batch = [tokens]
enc_batch = hf_tokenizer.encode_batch(batch, add_special_tokens=False)
print("[Method3] encode_batch.ids         →", enc_batch[0].ids)
print("[Method3] encode_batch.tokens      →", enc_batch[0].tokens)

# === STEP 10: Vocab sanity check ===
print("\n Vocab sanity check:")
vocab_set = hf_tokenizer.get_vocab()
for t in tokens:
    print(f"{t}: {'FOUND' if t in vocab_set else 'MISSING'}")
