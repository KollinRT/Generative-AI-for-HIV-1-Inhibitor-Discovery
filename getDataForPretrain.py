# TODO: NEW Rename getDataForPretrain.py... the finetune doesn't get filtered down...
# Add filter for criteria...

import pymysql
import pymysql.cursors
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import csv
import logging
import yaml
import os
import pandas as pd
from pandarallel import pandarallel

from prepare_dataset import convert_to_selfies 

# TODO: NEW Make some arg parser stuffs...
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

# make a parser for parallel processing
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--parallel", action="store_true", default=True, help="Use parallel processing"
# )
parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,          # <--  changed from True
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
# End of arg parser stuffs...
args = parser.parse_args()


# --------------------------------------------------------------------------- #
#  Load YAML config
# --------------------------------------------------------------------------- #
# now make yaml_file the --yaml argument
yaml_file = args.yaml
# yaml_file = './FinetuneSpecs.yml'
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


# Function to create a SQLite connection
# def create_db_connection():
#     # Connect to the SQLite database located at './db/chembl_34.db'
#     connection = sqlite3.connect('./db/chembl_34.db')
#     # This ensures that rows are returned as dictionaries
#     connection.row_factory = sqlite3.Row
#     return connection


# # # Function to execute SQL query
# def execute_sql_query(query):
#     connection = create_db_connection()
#     try:
#         with connection.cursor() as cursor:
#             cursor.execute(query)
#             results = cursor.fetchall()  # Fetch all results
#             return [result['canonical_smiles'] for result in results]
#     finally:
#         connection.close()

# # Function to execute SQL query
# def execute_sql_query(query):
#     connection = create_db_connection()
#     try:
#         with connection.cursor() as cursor:
#             cursor.execute(query)
#             results = cursor.fetchall()  # Fetch all results
#             return [dict(result)['canonical_smiles'] for result in results]
#     finally:
#         connection.close()

# def execute_sql_query(query):
#     connection = create_db_connection()
#     try:
#         cursor = connection.cursor()  # Create the cursor directly
#         cursor.execute(query)  # Execute the query
#         results = cursor.fetchall()  # Fetch all results
#         return [dict(result)['canonical_smiles'] for result in results]  # Process and return results
#     finally:
#         cursor.close()  # Close the cursor manually
#         connection.close()  # Ensure the connection is closed

# def execute_sql_query(query):
#     connection = create_db_connection()
#     try:
#         with connection.cursor() as cursor:
#             cursor.execute(query)
#             results = cursor.fetchall()  # Fetch all results
#             # Print the results to debug
#             for result in results:
#                 print(result)
#             return results
#     finally:
#         connection.close()




# chembl_34.compound_structures is where canonical_smiles is
# SELECT DISTINCT cs.canonical_smiles
# FROM compound_structures cs
# JOIN activities a ON cs.molregno = a.molregno
# JOIN assays ass ON a.assay_id = ass.assay_id
# WHERE ass.tid NOT IN (191, 12456) # not HIV-1 inhibs
# AND a.standard_type = 'IC50'; # for IC50 values


# Function to query ChEMBL database
# def query_chembl(excluded_tids):
#     query = f"""
#     SELECT DISTINCT canonical_smiles FROM activities
#     JOIN assays ON activities.assay_id = assays.assay_id
#     WHERE assays.tid NOT IN {tuple(excluded_tids)}
#     AND activities.standard_type = 'IC50'
#     """
#     return execute_sql_query(query)

# TODO: This is for the pretraining dataset....
# def query_chembl(excluded_tids):
#     query = f"""
#     SELECT DISTINCT cs.canonical_smiles
#     FROM compound_structures cs
#     JOIN activities a ON cs.molregno = a.molregno
#     JOIN assays ass ON a.assay_id = ass.assay_id
#     WHERE ass.tid NOT IN (191, 12456)
#     AND a.standard_type = 'IC50';
#     """
#     return execute_sql_query(query) # 954555
# TODO: If I can get these 954555 then make them filtered or what not then the model names save
# into the pretraining models to build them with different filters...

# def query_chembl(excluded_tids):
#     query = f"""
#     SELECT a.assay_id, a.doc_id, a.description,
#         t.assay_desc AS assay_type,
#         a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
#         a.chembl_id,
#         a.src_id,
#         CONCAT(act.value) AS IC50,
#         act.relation AS relation,
#         bs.site_name,
#         cs.canonical_smiles
#     FROM assays a
#     JOIN assay_type t ON a.assay_type = t.assay_type
#     JOIN activities act ON a.assay_id = act.assay_id
#     LEFT JOIN binding_sites bs ON a.tid = bs.tid
#     JOIN compound_structures cs ON act.molregno = cs.molregno
#     WHERE a.tid IN (191, 12456) AND
#         act.type LIKE 'IC50' AND
#         act.value IS NOT NULL AND
#         act.relation = '=' AND
#         (LOWER(a.assay_organism) LIKE 'human immunodeficiency virus%' OR
#         LOWER(a.assay_organism) LIKE 'hiv%')
#     """
#     # results = execute_sql_query(query)
#     # Print the columns of the first result to debug
#     # if results:
#     #     # print("Columns in the result:", results[0].keys())
#     # return results
#     return execute_sql_query(query) # 954555

# TODO: this is not needed for the pretrain data...... this should be query_pretrain_chembl......
# YASSSS... also finetune doesn't need data filtering...

# TODO: Fine tune is only one query... only the same query for all...
# pre-train is the one that differs! So...
# Maybe set a parent key that is different to indicate that it is finetuning?


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


# Finetuning dataset...
# Can get integrase or protease by filtering on site_name...
"""
use chembl_34;
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.value) AS IC50,
       act.relation AS relation,
       bs.site_name,
       cs.canonical_smiles
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id
LEFT JOIN binding_sites bs ON a.tid = bs.tid 
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE a.tid IN (191, 12456) AND
      act.type LIKE 'IC50' AND
      act.value IS NOT NULL AND
      act.relation = '=' AND
      (LOWER(a.assay_organism) LIKE 'human immunodeficiency virus%' OR
       LOWER(a.assay_organism) LIKE 'hiv%')
"""
# 4860 RVE (integrase) 1831 RVP (protease)
# TODO: this is not needed for the pretrain data...


# Function to draw SMILES
def draw_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), useSVG=True)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Function to save SMILES to CSV
# {key} so this should be like model_1, model_2, etc...
# I do not think so... this should be everything generic here still....
# for the model name should be post filtering...
# def save_smiles_to_csv(smiles_list, filename="vocab_smiles_data_TEST.csv"):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["canonical_smiles"])  # Write the header for the column
#         for smile in smiles_list: # I NEED ALL COLUMNS... IC50, site_name, etc...
#             writer.writerow([smile])  # Write each SMILES as a new row
# # TODO: NEW filename above should be the name of the model key in the yaml file and
# # you loop through it... So it should be within a loop...
def save_smiles_to_csv(smiles_list, filename="vocab_smiles_data_TEST.csv"):
    # TODO: This would need the key implemented in the filename...
    # Check if the list is not empty and get the keys from the first dictionary
    if smiles_list:
        headers = smiles_list[0].keys()
    else:
        headers = [
            "canonical_smiles",
            "IC50",
            "site_name",
        ]  # Default headers if the list is empty

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header for the columns
        for smile in smiles_list:
            writer.writerow(smile)  # Write each dictionary as a new row


# Example usage
excluded_tids = [191, 12456]
smiles = query_chembl(excluded_tids)
# print(smiles)
# print(smiles.columns)
# draw_smiles(smiles[:16])  # Draw the first 16 molecules
save_smiles_to_csv(smiles)  # Save all SMILES to CSV for pretrain...


# TODO: NEW do I want to define a lipinski filter here? If I do, it'd have a different filtering schema...
# It would need to only return the molecules that are RO5 compliant...

# I still need to define a function to filter from the compute properties....


# Load CSV file
df = pd.read_csv(
    "./vocab_smiles_data_TEST.csv"
)  # this is only containing `canonical_smiles`

# Ensure the 'canonical_smiles' column exists
if "canonical_smiles" not in df.columns:
    raise ValueError(
        "The input CSV file must contain a column named 'canonical_smiles'"
    )

# # --------------------------------------------------------------------------- #
# #  Initialize pandarallel – optional progress bar
# # --------------------------------------------------------------------------- #
if args.parallel:
    pandarallel.initialize(progress_bar=args.parallel)
# pandarallel.initialize(
#         nb_workers=os.cpu_count() - 1,
#         progress_bar=args.parallel)

# if args.parallel:
#     df["selfies"] = df["canonical_smiles"].parallel_apply(convert_to_selfies)
# else:
#     df["selfies"] = df["canonical_smiles"].apply(convert_to_selfies)

if args.parallel:
    properties_series = df.parallel_apply(compute_properties, axis=1)
else:
    properties_series = df.apply(compute_properties, axis=1)


# # Convert the Series to a DataFrame
properties_df = pd.DataFrame(properties_series.tolist())


# # --------------------------------------------------------------------------- #
# #  Convert SMILES → SELFIES and **add a new column**
# # --------------------------------------------------------------------------- #
# # We *don’t* overwrite the Series that contains the descriptors
# #   → we simply create a dedicated column.
# if args.parallel:
#     # df["selfies"] = df["canonical_smiles"].parallel_apply(convert_to_selfies)
#     # df.parallel_apply(lambda r: convert_to_selfies(r['canonical_smiles']), axis=1)
#     df["selfies"] = df["canonical_smiles"].parallel_apply(convert_to_selfies)  # try this after initialize
# else:
#     df["selfies"] = df["canonical_smiles"].apply(convert_to_selfies)

# --------------------------------------------------------------------------- #
#  Compute descriptors *after* selfies – we keep the raw SMILES
# --------------------------------------------------------------------------- #
# if args.parallel:
#     properties_series = df.parallel_apply(compute_properties, axis=1)
# else:
#     properties_series = df.apply(compute_properties, axis=1)

# # Convert the Series of dicts to a proper DataFrame
# props_df = pd.DataFrame(properties_series.tolist())

# --------------------------------------------------------------------------- #
#  Merge everything together
# --------------------------------------------------------------------------- #
# properties_df = pd.concat([df, props_df], axis=1)  # canonical_smiles, selfies, props...
# properties_df = properties_df.drop(columns=["canonical_smiles.1"])

# Print the DataFrame
print(properties_df)
print(type(properties_df))


# TODO: NEW I think this should be where the yaml file comes in for the models?
#                 this should be from the {key} from the yaml file...
#                 it should really be a filter function here based off of the yaml file....
properties_df.to_csv("./vocab_smiles_data_TEST_properties.csv", index=False)


# define a function to filter the properties_df based on criteria in the FineTuneSpecs.yml file...
# Load in the generic properties_df and then filter it based on the model key in the yaml file...
# This should be a function that is called after the properties_df is created...
# we do not change the properties_df, we just filter it and save it as a new file...
# Extract model names and their hyperparameters


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

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Iterate over each model and its hyperparameters
for model_name, filters in model_hyperparams.items():
    print(f"Processing model: {model_name}")
    for property_name, value in filters.items():
        print(f"  {property_name}: {value}")

    filter_properties(non_filter, model_name, filters)

    # FIX: IF WE STOP THIS RIGHT here we have the filtered CSV... but do I process into
    # SELFIES here or do I just do it to SMILES then let train_for_pretrain do the selfies conversion...?

    # Tokenize the filtered data using BPE tokenizer
    # bpe_save_to = f"./data/bpe_filter_{model_name}/"
    filtered_filename = f"model_name_{model_name}.csv"
    # FIX it should end here with the saving of the filtered_filename...
    """
    So this leaves me with the smiles and then the 5 descriptor columns...
    
    I need to then convert the smiles to selfies and then tokenize them 
    but this is in the train_for_pretrain.py file...?
    """
