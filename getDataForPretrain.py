import pymysql
import pymysql.cursors
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

import matplotlib.pyplot as plt
import csv
import logging
import yaml
import os
import pandas as pd
from pandarallel import pandarallel

import argparse

# Logging Config
logging.basicConfig(
    filename="conversion_errors.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# --------------------------------------------------------------------------- #
#  Argument parser
# --------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()

parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,
    help="Use parallel processing"
)
# now for yaml file
parser.add_argument(
    "--yaml",
    type=str,
    help="YAML file path for config stuffs",
    metavar="/path/to/hyperparameters/*.yml",
)
parser.add_argument(
    "--round",
    default="1",
    type=str,
    help="Round 1 or 2/3"
)
args = parser.parse_args()


# --------------------------------------------------------------------------- #
#  Load YAML config
# --------------------------------------------------------------------------- #
yaml_file = args.yaml

# Load the YAML file
with open(yaml_file, "r") as file:
    data = yaml.safe_load(file)

pymysql_info = data["pymysql_info"]

# --------------------------------------------------------------------------- #
#  Helper: DB connection
# --------------------------------------------------------------------------- #
# Function to establish connection to the database
def create_db_connection():
    connection = pymysql.connect(
        host=pymysql_info["host"],
        user=pymysql_info["user"],
        password=pymysql_info["password"],
        database=pymysql_info["database"],
        cursorclass=pymysql.cursors.DictCursor,
    )
    return connection
# --------------------------------------------------------------------------- #
#  SQL helpers
# --------------------------------------------------------------------------- #
def execute_sql_query(query):
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()  # Fetch all results
            # Extract only the 'canonical_smiles' column
            smiles_list = [
                result["canonical_smiles"]
                for result in results
                if "canonical_smiles" in result
            ]
            return smiles_list
    finally:
        connection.close()

# --------------------------------------------------------------------------- #
#  ChEMBL query - Arg passed for optional differentiation
# --------------------------------------------------------------------------- #
if args.round == "1":
    def query_chembl(excluded_tids):
        query = """
        SELECT DISTINCT cs.canonical_smiles
        FROM compound_structures cs
        JOIN activities a ON cs.molregno = a.molregno
        JOIN assays ass ON a.assay_id = ass.assay_id
        WHERE ass.tid NOT IN (191, 12456)
        AND a.standard_type = 'IC50'
        AND a.standard_value IS NOT NULL;
        """
        connection = create_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()  # Fetch all results
                return results  # Return the list of dictionaries
        finally:
            connection.close()
elif args.round == "2" or args.round == "3":
    def query_chembl(excluded_tids):
        query = """
        SELECT DISTINCT cs.canonical_smiles
        FROM compound_structures cs
        JOIN activities a ON cs.molregno = a.molregno
        JOIN assays ass ON a.assay_id = ass.assay_id
        WHERE ass.tid NOT IN (191, 12456)
        """
        connection = create_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()  # Fetch all results
                return results  # Return the list of dictionaries
        finally:
            connection.close()

#  ------------------------------------------------------------------ #
#  Compute descriptors 
#  ------------------------------------------------------------------ #
def compute_properties(row):
    from rdkit import Chem

    smiles = row["canonical_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        MW = Descriptors.MolWt(mol)
        numC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        chain_lengths = [len(fragment) for fragment in Chem.rdmolops.GetMolFrags(mol)]
        chain_length = max(chain_lengths) if chain_lengths else 0
        cLogP = Descriptors.MolLogP(mol)
        numRings = mol.GetRingInfo().NumRings()
        return {
            "canonical_smiles": smiles,
            "MW": MW,
            "numC": numC,
            "chain_length": chain_length,
            "cLogP": cLogP,
            "numRings": numRings,
        }
    else:
        return {
            "canonical_smiles": smiles,
            "MW": None,
            "numC": None,
            "chain_length": None,
            "cLogP": None,
            "numRings": None,
        }


def finetune_query_chembl(excluded_tids):
    query = """
    SELECT DISTINCT cs.canonical_smiles
    FROM compound_structures cs
    JOIN activities a ON cs.molregno = a.molregno
    JOIN assays ass ON a.assay_id = ass.assay_id
    WHERE ass.tid IN (191, 12456) # HIV-1 inhibs
    AND a.standard_type = 'IC50'; # for IC50 values
    """
    return execute_sql_query(query)  # 6877 for both

# Function to draw SMILES
def draw_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), useSVG=True)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_smiles_to_csv(smiles_list, filename="vocab_smiles_data_TEST.csv"):
    if smiles_list:
        headers = smiles_list[0].keys()
    else:
        headers = [
            "canonical_smiles",
            "IC50",
            "site_name",
        ]  
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header for the columns
        for smile in smiles_list:
            writer.writerow(smile)  # Write each dictionary as a new row


# Example usage
excluded_tids = [191, 12456]
smiles = query_chembl(excluded_tids)
save_smiles_to_csv(smiles)  # Save all SMILES to CSV for pretrain...

# Load CSV file
df = pd.read_csv(
    "./vocab_smiles_data_TEST.csv"
)  # this is only containing `canonical_smiles`

# Ensure the 'canonical_smiles' column exists
if "canonical_smiles" not in df.columns:
    raise ValueError(
        "The input CSV file must contain a column named 'canonical_smiles'"
    )

# --------------------------------------------------------------------------- #
#  Initialize pandarallel – optional progress bar
# --------------------------------------------------------------------------- #
if args.parallel:
    pandarallel.initialize(progress_bar=args.parallel)

# --------------------------------------------------------------------------- #
#  Compute descriptors
# --------------------------------------------------------------------------- #

if args.parallel:
    properties_series = df.parallel_apply(compute_properties, axis=1)
else:
    properties_series = df.apply(compute_properties, axis=1)


# Convert the Series to a DataFrame
properties_df = pd.DataFrame(properties_series.tolist())

# --------------------------------------------------------------------------- #
#  Merge everything together
# --------------------------------------------------------------------------- #
print(type(properties_df))

properties_df.to_csv("./vocab_smiles_data_TEST_properties.csv", index=False)

def filter_properties(properties_df, model_name, filters):
    # Filter the DataFrame based on the hyperparameters
    filtered_df = properties_df[
        (properties_df["MW"] <= filters["MW"])
        & (properties_df["numC"] <= filters["numC"])
        & (properties_df["chain_length"] <= filters["chain_length"])
        & (properties_df["cLogP"] <= filters["cLogP"])
        & (properties_df["numRings"] <= filters["numRings"])
    ]

    # Save the filtered DataFrame to a CSV file
    filename = f"model_name_{model_name}.csv"
    filtered_df.to_csv(filename, index=False)
    print(f"Filtered data saved to {filename}")


model_hyperparams = data["molecular_properties_to_filter"]  # PretrainSpecs.yml

# Load the initial DataFrame
non_filter = pd.read_csv("./vocab_smiles_data_TEST_properties.csv")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Iterate over each model and its hyperparameters
for model_name, filters in model_hyperparams.items():
    print(f"Processing model: {model_name}")
    for property_name, value in filters.items():
        print(f"  {property_name}: {value}")

    filter_properties(non_filter, model_name, filters)

    # bpe_save_to = f"./data/bpe_filter_{model_name}/"
    filtered_filename = f"model_name_{model_name}.csv"
