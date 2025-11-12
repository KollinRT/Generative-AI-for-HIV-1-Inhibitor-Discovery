import pandas as pd
from pandarallel import pandarallel
import selfies as sf
import logging
import argparse

from utils import load_hyperparameters

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
        return None  # Return None or a specific flag to indicate conversion failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_column",
        required=True,
        help="Column name of the SMILES dataset.",
    )
    parser.add_argument(
        "--hyperparameters_path",
        required=True,
        metavar="/path/to/hyperparameters/",
        help="Path to hyperparameters YAML file.",
    )
    args = parser.parse_args()

    pandarallel.initialize()

    hyperparameters = load_hyperparameters(args.hyperparameters_path)
    print(
        "Loaded hyperparameters:", hyperparameters
    )  # TODO: NEW 02/16/2025 figure out why BART is empty in combined_config... I THINK IT WORKS... ✓✓
    bart_hyperparameters = hyperparameters.get("BART", {})
    print("BART hyperparameters:", bart_hyperparameters)

    for key in bart_hyperparameters.keys():
        if key.startswith("skip_"):
            continue

        csv_file = f"model_name_{key}.csv"
        df = pd.read_csv(csv_file)
        df["selfies"] = df[f"{args.smiles_column}"].parallel_apply(convert_to_selfies)

        df.to_csv(f"{csv_file[:-4]}_selfies.csv")
        
