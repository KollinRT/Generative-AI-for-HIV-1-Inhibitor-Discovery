from typing import Optional

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
    """
    Args:
        smiles_string: str
            SMILES string to be converted to SELFIES
        # index:

    Returns:
        selfies_string || None: str

    """
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
    # path="data/smiles.csv", save_to="data/selfies_ready.csv"
    path: str = "data/smiles.csv",
    save_to: str = "data/selfies_ready.csv",
) -> Optional[str]:
    """
    Prepare a dataset for pretraining by loading SMILES strings,
    converting them to SELFIES, cleaning the data, and saving the result.
    Args:
        path : str, optional (default="data/smiles.csv")
            Path to the input CSV file containing SMILES strings.

        save_to : str, optional (default="data/selfies_ready.csv")
            Output path where the processed dataset will be saved.

    Returns:
        Optional[str]
            The path to the saved dataset if processing is completed,
            otherwise None.

    """
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
    """
    Args:
        selfies_df : pd.DataFrame
            Pandas DataFrame containing a column of SELFIES strings.

        save_to : str, optional (default="./data/selfies_subset.txt")
            File path where the resulting SELFIES text file will be saved.

        subset_size : int, optional (default=100000)
            Number of rows to sample if subsetting is enabled.

        do_subset : bool, optional (default=True)
            If True, a random subset of `subset_size` rows will be taken.
            If False, the entire DataFrame will be used.

    Returns:

    """
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
    """
    Args:
        read: str
            Directory for reading SELFIES strings.
        path: str
            File/directory for writing the list of selfies_alphabet in.
    Returns:
        N/A

    """
    df = pd.read_csv(read)
    selfies_array = df.selfies.to_numpy(copy=True)
    selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)
    with open(path, "w") as f:
        f.write(",".join(list(selfies_alphabet)))


def get_selfies_only(path, save_to):
    """
    Args:
        path : str
            Directory to read the SELFIES strings from.
        save_to : str
            Directory to save the SELFIES column only to.

    Returns:

    """
    df = pd.read_csv(path)
    selfies_column = df.selfies
    selfies_column.to_csv(save_to, index=False)
