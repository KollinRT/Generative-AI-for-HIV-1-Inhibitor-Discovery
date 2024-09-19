import pandas as pd
from pandarallel import pandarallel
import selfies as sf
import logging


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from os import mkdir


# def convert_to_selfies(smiles_col):  # returns selfies representation of smiles string. if there is no representation return smiles unchanged.
#     try:
#         return sf.encoder(smiles_col)
#     except sf.EncoderError:
#         print("EncoderError in Conversion")
#         return smiles_col

# Setup basic configuration for logging
logging.basicConfig(filename="conversion_errors.log", level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def convert_to_selfies(smiles_string, index=None):
    try:
        return sf.encoder(smiles_string)
    except sf.EncoderError:
        logging.info(f"EncoderError in Conversion at index {index} for SMILES: {smiles_string}")
        return None  # Return None or a specific flag to indicate conversion failure


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

def prepare_dataset_for_pretrain(path="data/smiles.csv", save_to="data/selfies_ready.csv"):
    smiles_data = pd.read_csv(path)
    pandarallel.initialize()
    smiles_data["selfies"] = smiles_data["canonical_smiles"].parallel_apply(convert_to_selfies)
    smiles_data.drop(smiles_data[smiles_data.selfies.isnull()].index, inplace=True)
    smiles_data.drop(columns=["canonical_smiles"], inplace=True)
    smiles_data.to_csv(save_to, index=False)

def create_selfies_file(selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000, do_subset=True):

    selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling

    if do_subset:
        selfies_subset = selfies_df.selfies[:subset_size]
    else:
        selfies_subset = selfies_df.selfies
    selfies_subset = selfies_subset.to_frame()
    selfies_subset["selfies"].to_csv(save_to, index=False, header=False)

# def get_selfies_alphabet(read="./data.csv", path="./data/selfies_alphabet.csv"):
#     df = pd.read_csv(read)
#     selfies_array = df.selfies.to_numpy(copy=True)
#     selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)
#
#     with open(path, "w") as f:
#         f.write(",".join(list(selfies_alphabet)))

def get_selfies_alphabet(read="data/selfies_ready.csv", path="data/selfies_alphabet.csv"):
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


# def bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/"):
#     """
#     BPE tokenizer
#     :param path: Path to get data from
#     :param save_to: Path to save the folder of the tokenizer
#     :return: N/A
#     """
# 	try:
# 		mkdir(save_to)
# 	except FileExistsError:
# 		pass
#
# 	tokenizer = Tokenizer(BPE(unk_token="<unk>"))
#
# 	tokenizer.pre_tokenizer = Split(pattern=Regex("\[|\]"), behavior="removed")
#
# 	tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1", special_tokens=[("<s>", 1), ("</s>", 2)],)
#
# 	trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
# 	tokenizer.train(files=[path], trainer=trainer)
#
# 	tokenizer.save(save_to + "/bpe.json", pretty=True)
# 	tokenizer.model.save(save_to)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

# TODO: Should be more improved to more comprehensively deals with SELFIES than the previous BPE tokenizer.
# def bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/"):
#     """
#     BPE tokenizer configured for SELFIES data and suitable for generative tasks using models like BART.
#
#     :param path: Path to get data from.
#     :param save_to: Path to save the folder of the tokenizer.
#     :return: None
#     """
#     # Ensure the directory exists
#     try:
#         mkdir(save_to)
#     except FileExistsError:
#         pass
#
#     # Create a tokenizer instance with BPE
#     tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
#
#     # Setup a pre-tokenizer to correctly tokenize SELFIES without losing structural brackets
#     # tokenizer.pre_tokenizer = pre_tokenizers.RegexSplit(pattern="(\\[[^\\[\\]]+\\])", behavior="isolated") # (\[[^\[\]]+\])
#     # tokenizer.pre_tokenizer = Split(pattern=Regex("(\\[[^\\[\\]]+\\])"), behavior="isolated") # delimiter not removed; keep []
#     tokenizer.pre_tokenizer = pre_tokenizers.Split(
#         pattern=Regex(r"(\[[^\[\]]*\])|(\][^\[\]]*\[)|(\][^\[\]]*$)|(^[^\[\]]*\[)"),
#         behavior="isolated"  # or "removed", depending on your requirements
#     )
#
#     # Define a post-processor to add start and end tokens necessary for generative models
#     # tokenizer.post_processor = processors.TemplateProcessing(
#     #     single="<s> $A </s>",
#     #     pair="<s> $A </s> $B:1 </s>:1",
#     #     special_tokens=[
#     #         ("<s>", tokenizer.token_to_id("<s>")),  # Ensure <s> token is recognized
#     #         ("</s>", tokenizer.token_to_id("</s>"))  # Ensure </s> token is recognized
#     #     ]
#     # )
#     tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1", special_tokens=[("<s>", 1), ("</s>", 2)],)
#
#     # Configure the trainer with specific special tokens, including those needed for generative tasks
#     trainer = trainers.BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
#
#     # Train the tokenizer on the specified file
#     tokenizer.train(files=[path], trainer=trainer)
#
#     # Save the tokenizer and model to the specified directory
#     tokenizer.save(save_to + "/bpe.json", pretty=True)
#     tokenizer.model.save(save_to)
#


def bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/"):
    """
    BPE tokenizer configured for SELFIES data and suitable for generative tasks using models like BART.

    :param path: Path to get data from.
    :param save_to: Path to save the folder of the tokenizer.
    :return: None
    """
    # Ensure the directory exists
    try:
        mkdir(save_to)
    except FileExistsError:
        pass

    # Create a tokenizer instance with BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"(\[[^\[\]]*\])|(\][^\[\]]*\[)|(\][^\[\]]*$)|(^[^\[\]]*\[)"),
        behavior="isolated"  # or "removed", depending on your requirements
    )

    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1", special_tokens=[("<s>", 1), ("</s>", 2)],)

    # Configure the trainer with specific special tokens, including those needed for generative tasks
    trainer = trainers.BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])

    # Train the tokenizer on the specified file
    tokenizer.train(files=[path], trainer=trainer)

    # Save the tokenizer and model to the specified directory
    tokenizer.save(save_to + "/bpe.json", pretty=True)
    tokenizer.model.save(save_to)

# check with vocab

