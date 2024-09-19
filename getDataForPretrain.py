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



# TODO: NEW Make some arg parser stuffs...
import argparse
# make a parser for parallel processing
parser = argparse.ArgumentParser()
parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing')
# now for yaml file
parser.add_argument('--yaml', type=str, help='YAML file path for config stuffs', metavar="/path/to/hyperparameters/*.yml")
# Now do this for

# End of arg parser stuffs...
args = parser.parse_args()



# Configure a yaml file for config stuffs...

# now make yaml_file the --yaml argument
yaml_file = args.yaml
# yaml_file = './FinetuneSpecs.yml'
# Load the YAML file
with open(yaml_file, 'r') as file:
    data = yaml.safe_load(file)

pymysql_info=data['pymysql_info']

# Function to establish connection to the database
def create_db_connection():
    connection = pymysql.connect(host=pymysql_info['host'],
                                 user=pymysql_info['user'],
                                 password=pymysql_info['password'],
                                 database=pymysql_info['database'],
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

# Function to execute SQL query
def execute_sql_query(query):
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()  # Fetch all results
            return [result['canonical_smiles'] for result in results]
    finally:
        connection.close()
 
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
def query_chembl(excluded_tids):
    query = f"""
    SELECT DISTINCT cs.canonical_smiles
    FROM compound_structures cs
    JOIN activities a ON cs.molregno = a.molregno
    JOIN assays ass ON a.assay_id = ass.assay_id
    WHERE ass.tid NOT IN (191, 12456) # not HIV-1 inhibs
    AND a.standard_type = 'IC50'; # for IC50 values
    """
    return execute_sql_query(query) # 954555
# TODO: If I can get these 954555 then make them filtered or what not then the model names save
# into the pretraining models to build them with different filters...



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

# TODO: New write function to query using RDKit to get molecules that fit some
# set of parameters less than...

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

# Function to draw SMILES
def draw_smiles(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), useSVG=True)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Function to save SMILES to CSV
                                                          #{key} so this should be like model_1, model_2, etc...
                                                          # I do not think so... this should be everything generic here still....
                                                          # for the model name should be post filtering...
def save_smiles_to_csv(smiles_list, filename="vocab_smiles_data_TEST.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["canonical_smiles"])  # Write the header
        for smile in smiles_list:
            writer.writerow([smile])  # Write each SMILES as a new row
# TODO: NEW filename above should be the name of the model key in the yaml file and 
# you loop through it... So it should be within a loop...


# Example usage
excluded_tids = [191, 12456]
smiles = query_chembl(excluded_tids)
# draw_smiles(smiles[:16])  # Draw the first 16 molecules
save_smiles_to_csv(smiles)  # Save all SMILES to CSV for pretrain...

# If I want to pretrain on filtered data, I need to be able to do this here...

# Function to compute properties
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        MW = Descriptors.MolWt(mol)
        numC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        chain_lengths = [len(fragment) for fragment in Chem.rdmolops.GetMolFrags(mol)]
        chain_length = max(chain_lengths) if chain_lengths else 0
        cLogP = Descriptors.MolLogP(mol)
        numRings = mol.GetRingInfo().NumRings()
        return {'SMILES': smiles, 'MW': MW, 'numC': numC, 'chain_length': chain_length, 'cLogP': cLogP, 'numRings': numRings}
    else:
        return {'SMILES': smiles, 'MW': None, 'numC': None, 'chain_length': None, 'cLogP': None, 'numRings': None}

# TODO: NEW do I want to define a lipinski filter here? If I do, it'd have a different filtering schema...
# It would need to only return the molecules that are RO5 compliant...

# I still need to define a function to filter from the compute properties....


# Load CSV file
df = pd.read_csv('./vocab_smiles_data_TEST.csv')

# Ensure the 'canonical_smiles' column exists
if 'canonical_smiles' not in df.columns:
    raise ValueError("The input CSV file must contain a column named 'canonical_smiles'")


# make if statement for parallel processing
if args.parallel == True:
    # If parallel processing is available, use it
    pandarallel.initialize(progress_bar=True)
    properties_df = df['canonical_smiles'].parallel_apply(compute_properties)
    
    properties_df = pd.DataFrame(properties_df.tolist())
else:
    # Otherwise, use sequential processing
    properties_df = df['canonical_smiles'].apply(compute_properties)
    # properties = [compute_properties(smiles) for smiles in df['canonical_smiles']]
    properties_df = pd.DataFrame(properties)

    


# df.apply(func)
# df.parallel_apply(compute_properties(df['canonical_smiles']), axis=1)


# Sequentially
# Compute properties for all SMILES
# properties = [compute_properties(smiles) for smiles in df['canonical_smiles']]

# Create DataFrame from computed properties
# properties_df = pd.DataFrame(properties)


# Print the DataFrame
print(properties_df)


# TODO: NEW I think this should be where the yaml file comes in for the models?
#                 this should be from the {key} from the yaml file...
#                 it should really be a filter function here based off of the yaml file....
properties_df.to_csv('./vocab_smiles_data_TEST_properties.csv', index=False)



# define a function to filter the properties_df based on criteria in the FineTuneSpecs.yml file...
# Load in the generic properties_df and then filter it based on the model key in the yaml file...
# This should be a function that is called after the properties_df is created...
# we do not change the properties_df, we just filter it and save it as a new file...
# Extract model names and their hyperparameters

def filter_properties(properties_df, model_name, filters):
    # Filter the DataFrame based on the hyperparameters
    filtered_df = properties_df[
        (properties_df['MW'] <= filters['MW']) &
        (properties_df['numC'] <= filters['numC']) &
        (properties_df['chain_length'] <= filters['chain_length']) &
        (properties_df['cLogP'] <= filters['cLogP']) &
        (properties_df['numRings'] <= filters['numRings'])
    ]
    
    # Save the filtered DataFrame to a CSV file
    filename = f"model_name_{model_name}.csv"
    filtered_df.to_csv(filename, index=False)
    print(f"Filtered data saved to {filename}")


model_hyperparams = data['molecular_properties_to_filter']
 
# Load the initial DataFrame
non_filter = pd.read_csv("/home/kollin/Desktop/ThesisCode/ThesisCode/vocab_smiles_data_TEST_properties.csv")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Iterate over each model and its hyperparameters
for model_name, filters in model_hyperparams.items():
    print(f"Processing model: {model_name}")
    for property_name, value in filters.items():
        print(f"  {property_name}: {value}")
    
    # Filter properties and save the filtered DataFrames
    filter_properties(non_filter, model_name, filters)
    
    # Tokenize the filtered data using BPE tokenizer
    bpe_save_to = f"./data/bpe_filter_{model_name}/"
    filtered_filename = f"model_name_{model_name}.csv"
    bpe_tokenizer(filtered_filename, save_to=bpe_save_to)
    # FIX maybe tokenize after dropping the SMILES column... and all the others besides selfies...
    
    # Load the filtered DataFrame
    df = pd.read_csv(filtered_filename)
    
    # For SMILES to SELFIES, start by making a new column by copying the canonical SMILES
    df["selfies"] = df["SMILES"]
    
    # Convert to SELFIES
    df.selfies = df.selfies.parallel_apply(convert_to_selfies)
    
    print(df.columns)
    print(df.selfies.head(100))
    
    # Remove molecules that are not converted
    df.drop(df[df.SMILES == df.selfies].index, inplace=True)
    
    # Drop the canonical_smiles representation
    df.drop(columns=["SMILES"], inplace=True)
    print(f"second df.columns: {df.columns}")
    print(f"second len(df.selfies): {len(df.selfies)}")
    
    
    
    # Drop all columns except for selfies
    df = df[['selfies']]

    # Remove the first row
    df = df.iloc[1:]
    # TODO: Now it would be ready for the BPE tokenizer...


    # Save to a .csv file
    selfies_save_to = f"./data/filtered_selfies_{model_name}.csv"
    df.to_csv(selfies_save_to, index=False)
    df = pd.read_csv(selfies_save_to, index=False, header=None)
    bpe_tokenizer(selfies_save_to, save_to=bpe_save_to)

    
    print(f"SELFIES data saved to {selfies_save_to}")



# TODO: So then I would load the properties_df just the canonical smiles and then do the vocabulary building
# Which would be based off the key of the model... So I would have to loop through the keys of the model


# # TODO: Build in a function to perform the vocabulary building for the pretraining dataset...
# # This is from the file: JustTestSelfiesCLEAN_RUN.py

# from prepare_dataset import prepare_dataset_for_pretrain, get_selfies_alphabet
# import pandas as pd
# from pandarallel import pandarallel
# import selfies as sf
# from prepare_dataset import bpe_tokenizer, get_selfies_only

# def build_selfies_vocab(selfies_list):
#     unique_symbols = set()
#     # Ensure that selfies is a string before processing
#     for selfies in filter(lambda x: isinstance(x, str), selfies_list):
#         symbols = sf.split_selfies(selfies)
#         unique_symbols.update(symbols)
#     return {symbol: i for i, symbol in enumerate(unique_symbols)}


# if __name__ == '__main__':
#     non_filter = pd.read_csv("/home/kollin/Downloads/Thesis_04302024/ThesisProject_SentToRun/ChEMBL34_druglike_activity.csv")
#     bpe_tokenizer("./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv", save_to="./data/bpe_non/")

#     # # For SMILES to SELFIES, start by making a new column by copying the canonical SMILES
#     # non_filter["selfies"] = non_filter["canonical_smiles"]
#     #
#     # # Convert to SELFIES
#     # non_filter.selfies = non_filter.selfies.parallel_apply(sf.encoder)
#     #
#     # # Remove molecules that are not converted
#     # non_filter.drop(non_filter[non_filter.canonical_smiles == non_filter.selfies].index, inplace=True)
#     # # Drop the canonical_smiles representation
#     # non_filter.drop(columns=["canonical_smiles"], inplace=True)
#     # # Save to a .csv file
#     # non_filter.to_csv(save_to, index=False)

#     df = pd.read_csv("./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv")
#     bpe_tokenizer("./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv", save_to="./data/bpe_filter/")