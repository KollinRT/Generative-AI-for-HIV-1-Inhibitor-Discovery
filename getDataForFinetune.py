# TODO: NEW Rename getDataForPretrain.py... the finetune doesn't get filtered down...
# Add filter for criteria...

import pymysql
import pymysql.cursors
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
# from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import csv
import yaml
import pandas as pd
from pandarallel import pandarallel

from prepare_dataset import prepare_dataset_for_pretrain, get_selfies_alphabet
import selfies as sf # TODO: Figure out the error handling for this... It is somewhere in my code...
from prepare_dataset import bpe_tokenizer, get_selfies_only, convert_to_selfies

import sqlite3

# TODO: NEW Make some arg parser stuffs...
import argparse
# make a parser for parallel processing
parser = argparse.ArgumentParser()
parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing')
# now for yaml file
parser.add_argument('--yaml', type=str, help='YAML file path for config stuffs', metavar="/path/to/hyperparameters/*.yml")
# Now do this for pretrain/finetune
parser.add_argument('--mode', type=str, help='Decide between pretrain or finetune')

# End of arg parser stuffs...
args = parser.parse_args()



# Configure a yaml file for config stuffs...

# now make yaml_file the --yaml argument
yaml_file = args.yaml
# yaml_file = './FinetuneSpecs.yml'
# Load the YAML file

with open(yaml_file, 'r') as file:
    data  = yaml.safe_load(file)

pymysql_info=data['pymysql_info']

# Function to establish connection to the database
def create_db_connection():
    connection = pymysql.connect(host=pymysql_info['host'],
                                 user=pymysql_info['user'],
                                 password=pymysql_info['password'],
                                 database=pymysql_info['database'],
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

def query_chembl(excluded_tids):
    query = f"""
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
        LOWER(a.assay_organism) LIKE 'hiv%');
    """
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()  # Fetch all results
            return results  # Return the list of dictionaries
    finally:
        connection.close()

def compute_properties(row):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    smiles = row['canonical_smiles']
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        MW = Descriptors.MolWt(mol)
        numC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        chain_lengths = [len(fragment) for fragment in Chem.rdmolops.GetMolFrags(mol)]
        chain_length = max(chain_lengths) if chain_lengths else 0
        cLogP = Descriptors.MolLogP(mol)
        numRings = mol.GetRingInfo().NumRings()
        return {
            'canonical_smiles': smiles,
            'MW': MW,
            'numC': numC,
            'chain_length': chain_length,
            'cLogP': cLogP,
            'numRings': numRings,
            'IC50': row['IC50'],
            'site_name': row['site_name'] # TODO: this is not needed for the pretrain data...
        }
    else:
        return {
            'canonical_smiles': smiles,
            'MW': None,
            'numC': None,
            'chain_length': None,
            'cLogP': None,
            'numRings': None,
            'IC50': row['IC50'],
            'site_name': row['site_name']
        }

def execute_sql_query(query):
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()  # Fetch all results
            # Extract only the 'canonical_smiles' column
            smiles_list = [result['canonical_smiles'] for result in results if 'canonical_smiles' in result]
            return smiles_list
    finally:
        connection.close()
 

def finetune_query_chembl(excluded_tids):
    query = f"""
    SELECT DISTINCT cs.canonical_smiles
    FROM compound_structures cs
    JOIN activities a ON cs.molregno = a.molregno
    JOIN assays ass ON a.assay_id = ass.assay_id
    WHERE ass.tid IN (191, 12456) # HIV-1 inhibs
    AND a.standard_type = 'IC50'; # for IC50 values
    """
    return execute_sql_query(query) # 6877 for both


# Function to draw SMILES
def draw_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), useSVG=True)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def save_smiles_to_csv(smiles_list, filename="smiles_finetune_data.csv"):
    # TODO: This would need the key implemented in the filename...
    # Check if the list is not empty and get the keys from the first dictionary
    if smiles_list:
        headers = smiles_list[0].keys()
    else:
        headers = ["canonical_smiles", "IC50", "site_name"]  # Default headers if the list is empty

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header for the columns
        for smile in smiles_list:
            writer.writerow(smile)  # Write each dictionary as a new row

# Example usage
excluded_tids = [191, 12456]
smiles = query_chembl(excluded_tids)
save_smiles_to_csv(smiles)  # Save all SMILES to CSV for pretrain...
print(f"Saved {len(smiles)} SMILES to smiles_finetune_data.csv")
# Load CSV file
df = pd.read_csv('./smiles_finetune_data.csv') # this is only containing `canonical_smiles`

# Ensure the 'canonical_smiles' column exists
if 'canonical_smiles' not in df.columns:
    raise ValueError("The input CSV file must contain a column named 'canonical_smiles'")


pandarallel.initialize(progress_bar=True)
if args.parallel:
    properties_series = df.parallel_apply(compute_properties, axis=1)
else:
    properties_series = df.apply(compute_properties, axis=1)


# Convert the Series to a DataFrame
properties_df = pd.DataFrame(properties_series.tolist())


# Print the DataFrame
print(properties_df)
print(type(properties_df))

properties_df.to_csv('./data/smiles_finetune_data_properties.csv', index=False)
 
# Load the initial DataFrame
non_filter = pd.read_csv("./data/smiles_finetune_data_properties.csv")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pandarallel.initialize()
non_filter["selfies"] = non_filter["canonical_smiles"].parallel_apply(convert_to_selfies)
non_filter.drop(non_filter[non_filter.selfies.isnull()].index, inplace=True)
print(non_filter.columns)  # DEBUG
print(f"smiles_data.columns post-drop: {non_filter.columns}")
non_filter.to_csv(f"./data/smiles_finetune_data_properties_selfies.csv", index=False)  # TODO: index=False, header=False for sure...

# print(f"saving properties df to: ./data/smiles_finetune_data_properties.csv")