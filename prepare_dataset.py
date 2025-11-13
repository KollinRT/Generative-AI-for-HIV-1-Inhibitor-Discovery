import pandas as pd
from pandarallel import pandarallel
import selfies as sf
import logging

import os
from tokenizers import models, pre_tokenizers, trainers

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

# Setup basic configuration for logging
logging.basicConfig(
    filename="conversion_errors.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def convert_to_selfies(smiles_string, index=None):
    try:
        return sf.encoder(smiles_string)
    except sf.EncoderError:
        logging.info(
            f"EncoderError in Conversion at index {index} for SMILES: {smiles_string}"
        )
        return None 

# Inspired by https://github.com/HUBioDataLab/SELFormer/blob/main/prepare_pretraining_data.py
# def prepare_dataset_for_pretrain(path="data/smiles.txt", save_to="data/selfies_ready.csv"):
#     # smiles_data = pd.read_csv(path, sep="\t")
#     smiles_data = pd.read_csv(path)
#
#     # Initialize parallel application for formatting SELFIES data
#     pandarallel.initialize()
#
#     # For SMILES to SELFIES, start by making a new column by copying the canonical SMILES
#     smiles_data["selfies"] = smiles_data["canonical_smiles"]
#
#     # Convert to SELFIES
#     smiles_data.selfies = smiles_data.selfies.parallel_apply(convert_to_selfies)
#
#     # Remove molecules that are not converted
#     smiles_data.drop(smiles_data[smiles_data.canonical_smiles == smiles_data.selfies].index, inplace=True)
#     # Drop the canonical_smiles representation
#     smiles_data.drop(columns=["canonical_smiles"], inplace=True)
#     # Save to a .csv file
#     smiles_data.to_csv(save_to, index=False)


def prepare_dataset_for_pretrain(
    path="data/smiles.csv", save_to="data/selfies_ready.csv"
):
    smiles_data = pd.read_csv(path)
    print(f"smiles_data.columns: {smiles_data.columns}")  # DEBUG
    # print(smiles_data.data.head())
    pandarallel.initialize()
    smiles_data["selfies"] = smiles_data["canonical_smiles"].parallel_apply(
        convert_to_selfies
    )
    smiles_data.drop(smiles_data[smiles_data.selfies.isnull()].index, inplace=True)
    print(smiles_data.columns)  # DEBUG
    smiles_data.drop(
        columns=["canonical_smiles"], inplace=True
    )  # TODO: Drop all besides selfies...
    print(f"smiles_data.columns post-drop: {smiles_data.columns}")
    smiles_data.to_csv(
        save_to, index=False
    )  # TODO: index=False, header=False for sure...


def create_selfies_file(
    selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000, do_subset=True
):
    selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling

    if do_subset:
        selfies_subset = selfies_df.selfies[:subset_size]
    else:
        selfies_subset = selfies_df.selfies
    selfies_subset = selfies_subset.to_frame()  #
    selfies_subset["selfies"].to_csv(save_to, index=False, header=False)
    print("SELFIES_SUBSET here")
    print(selfies_subset)


def get_selfies_alphabet(
    read="data/selfies_ready.csv", path="data/selfies_alphabet.csv"
):
    df = pd.read_csv(read)
    selfies_array = df.selfies.to_numpy(copy=True)
    selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)
    with open(path, "w") as f:
        f.write(",".join(list(selfies_alphabet)))


# TODO: Not sure if needed
def get_selfies_only(path, save_to):
    df = pd.read_csv(path)
    selfies_column = df.selfies
    selfies_column.to_csv(save_to, index=False)




# def bpe_tokenizer(
#     path: str = "./data/selfies_subset.txt", save_to: str = "./data/bpe/"
# ) -> None:
#     """
#     BPE tokenizer configured for SELFIES strings,
#     isolating each [X] symbol as its own token and
#     ready for encoder–decoder models (e.g. BART).
#     """
#     # 1) make sure the output directory exists
#     os.makedirs(save_to, exist_ok=True)

#     # 2) initialize an “empty” BPE model
#     tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

#     # 3) pre-tokenizer: split out every “[...]” as its own piece
#     #    either of these two variants will work:

#     # Option A: pass a raw regex string
#     tokenizer.pre_tokenizer = pre_tokenizers.Split(
#         pattern=r"\[[^\]]+\]",
#         behavior="isolated",
#     )

#     # Option B: compile with Python’s `re` and pass that
#     # tokenizer.pre_tokenizer = pre_tokenizers.Split(
#     #     pattern=re.compile(r"\[[^\]]+\]"),
#     #     behavior="isolated",
#     # )

#     # 4) post-processor: wrap sequences in <s>…</s>, handle pairs
#     tokenizer.post_processor = TemplateProcessing(
#         single="<s> $A </s>",
#         pair="<s> $A </s> $B:1 </s>:1",
#         special_tokens=[("<s>", 1), ("</s>", 2)],
#     )

#     # 5) trainer: remind it about all the “special” tokens you’ll use
#     trainer = trainers.BpeTrainer(
#         special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
#     )

#     # 6) train on your SELFIES file
#     tokenizer.train(files=[path], trainer=trainer)

#     # 7) save both the JSON “tokenizer” and the raw BPE model files
#     tokenizer.save(os.path.join(save_to, "bpe.json"), pretty=True)
#     tokenizer.model.save(save_to)

#     print(f"✅ BPE tokenizer written to {save_to!r}")
