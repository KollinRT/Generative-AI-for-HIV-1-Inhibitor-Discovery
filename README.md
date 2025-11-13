## Tech Specs
Intel Core i9-13900KF Processor (24 core (8P/16E)/32 Threads)  
64 GB RAM 4800 MHz  
RTX 4090 24GB VRAM  

# How to run
## How to configure the input files...

In order for set up to run, we need to first download data for [ChEMBL_35](https://chembl.gitbook.io/chembl-interface-documentation/downloads), which we would download the ChEMBLdb and the database format of your choice.

To configure the `FinetuneSpecs.yml` file, we need to figure out two things.

```yaml
pymysql_info:
    host: '127.0.0.1'
    user: 'dev'
    password: 'devpass'
    database: 'chembl_35'
molecular_properties_to_filter:
    model_1:
        MW: 500
        numC: 6
        chain_length: 6
        cLogP: 5
        numRings: 3
    model_2:
        MW: 400
        numC: 5
        chain_length: 5
        cLogP: 4
        numRings: 2
    model_3:
        MW: 600
        numC: 7
        chain_length: 7
        cLogP: 6
        numRings: 4
```

The generic base model `combined_config.yml` looks like the following file. This is generated from the `MakeDefaultHyperparams.py` file.

```yaml
BART:
  model_base:
    HIDDEN_SIZE: 768
    TRAIN_BATCH_SIZE: 16
    VALID_BATCH_SIZE: 8
    TRAIN_EPOCHS: 100
    LEARNING_RATE: 5.0e-05
    WEIGHT_DECAY: 0.01
    MAX_LEN: 128
    ENCODER_LAYERS: 12
    DECODER_LAYERS: 12
    NUM_ENCODER_ATTENTION_HEADS: 12
    NUM_DECODER_ATTENTION_HEADS: 12
    ENCODER_FFN_DIM: 3072
    DECODER_FFN_DIM: 3072
    MAX_POSITION_EMBEDDINGS: 514
    NUM_ATTENTION_HEADS: 12
    NUM_HIDDEN_LAYERS: 8
    TYPE_VOCAB_SIZE: 1
    optimizer: adam
    criterion: crossentropy
    early_stopping_toggle: true
    early_stopping_threshold: 5.0e-05
    early_stopping_patience: 5
    lr_sched: null
```
Which is what is modified to build up the models described in the project.

`FinetuneSpecs.yml` should look like where we just give the database information
```
pymysql_info:
    host: '127.0.0.1'
    user: 'dev'
    password: 'devpass'
    database: 'chembl_35'
```
Then running
```bash
python3 getDataForFinetune.py --yaml "./FinetuneSpecs.yml"
```
returns a `.csv` file with 
`canonical_smiles,MW,numC,chain_length,cLogP,numRings,IC50,site_name`.

#### Definitive Workflow make into a bash script
Build the `Dockerfile` and run the container.

```bash
# Round 1
python getDataForPretrain.py --parallel --yaml "./PretrainSpecs.yml"
## If not-parallel
python getDataForPretrain.py --yaml "./PretrainSpecs.yml"

# To generate `combined_config.yml`
python3 MakeDefaultHyperparams.py --file="./PretrainSpecs.yml"

# Convert to SELFIES
python addSelfiesDescription.py --smiles_column="canonical_smiles" --hyperparameters_path="./combined_config.yml"

python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml"

# Round 2/3
python3 getDataForPretrain.py --yaml="./PretrainSpecs.yml" --round="2"

python addSelfiesDescription.py --smiles_column="canonical_smiles" --hyperparameters_path="./combined_config.yml"

python combine_cleanup_for_PT_token.py

python filter_embedding.py

# # Can not be done locally with only 64 gb RAM. Needs about 300gb 
# python3 prepro_ClustFing.py 

python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml"

# Round 3
python3 train_for_pretrain.py --hyperparameters_path="./combined_config_steps.yml"


# Finetuning
python3 train_for_finetune.py --hyperparameters_path="./combined_config_WIP_FT.yml"
```

# Steps
Build the Dockerfile and run the container.

## How to acquire the data used in this

### Download ChEMBL35 or newer from the portal
[ChEMBL Downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)

Create the `chembl_35` database. 
```
CREATE DATABASE IF NOT EXISTS chembl_35;
```
Then fill in the database via the following command utilized with our Docker container. This populates the DB needed to work further on.
```
$> mysql -udev -pdevpass -h127.0.0.1 -P3306 chembl_35 < chembl_35_mysql.dmp
```    
This will populate the `chembl_35` database inside our container.    

## Download Druglike Molecules from ZINC 15
[ZINC15 Tranches](https://zinc15.docking.org/tranches/home/)

Click the 3x3 dots button <img src="./images/ZINC15_dots.png" width="25" height="25"> and click the Druglike filter and click off of 5 on left and 500 on top. This will get the data in download that can be downloaded with a script via WGET. Run that script in the data directory to generate the data.

or use the existing [File in repo](./scripts/download_zinc.sh)

```bash
sh /scripts/download_zinc.sh
```
From the directory you want the data stored in. I suggest `/data`.
Now, let the whole file directory download. Much like the ChEMBL DB population, this might take several hours so please set time aside to accommodate this.

### Get Round 2 Data 10M
Once the data is in the directory needed we can run the script `combine_zinc15.sh` which will combine all the ZINC15 `*.smi` files so that we can then sample 10M for Round 2.

```
awk 'FNR==1 && NR!=1 {next} {print}' $(find . -type f -name "*.smi") > zinc15_all_raw.smi
```
The top one is quicker but is also grabbing the header multiple times. The bottom one times longer but is not.
Then one could run 

`WIP_Thesis/scripts/Round2/train_word_level_tokenizer_round2.py` is the file used to create the tokenizer for Round 2. Just need to now find the file that SELFIES the other files. Process the ChEMBL 35 dataset by
```
shuf -n 10000000 zinc15_all_raw.smi > zinc15_sampled_10M.smi
```
to get 10M molecules needed for Round 2. This gives me the molecules with `smiles` and `zinc_id`. Need to check.

Then once the data is collected and translated to SELFIES via `process_10M_parallel.py`, the finetune data is obtained via `getDataForFinetune.py` and the pretraining dataset is obtained via `getDataForPretrain.py` then one can filter the combine the data via `python combine_cleanup_for_PT_token.py` and then filter by embedding length via `filter_embedding.py` then we are ready for tokenization. Tokenization is carried out via the `train_word_level_tokenizer_round2.py` file in scripts/Round2. 

Then from here we can combine the 10M sampled with the 2.3M pretrained molecules to then do fingerprints and clustering. However, this will have to be done on a larger system as more RAM is needed. More work could be done into finding or creating a more efficient scheme for memory-efficient big data fingerprint clustering but this was not needed at the time of experimentation since a higher RAM memory machine could be obtained for fairly cheap. There are more recent options for clustering that seem to fare well such as [BitBIRCH](https://github.com/mqcomplab/bitbirch) but this publisted after the work was already carried out and didn't seem to impact appreciably.

### Get Round 3 Data ~860M Molecules
Getting the data for Round 3 is done through downloading the data from the ZINC15 tranches as described in the section above. Then take the data and mix them all into one data source just like in Round 2.

Then we need to work to process the ~860M ZINC15 druglike molecules:
`process_in_chunk.py` is utilized to make 860 1M molecules file to then process into a parquet file. This parquet file will interact with Dask in order to utilize the big data streaming for Round 3.

To generate the parquet file for big data we need to move the `zinc15_all_raw.smi` to the directory with the script and run `process_in_chunk.py` to chunk and translate all the molecules into SELFIES. Next is to run the `dask_combine_fixed.py` to combine all the `.csv` chunk files into a parquet directory to work with our streaming global steps training regime. 

## Build the Docker image
```
docker build -t gen_ai_hiv .
```
## Run the Docker container
```
docker run -it --rm \
	--shm-size=2g \
    -v "$(pwd)"/data/mysql_data:/var/lib/mysql \
    -v "$(pwd)/data":/data \
    -v "$(pwd)":/app \
    -p 3307:3306 \
    gen_ai_hiv
```
## Run with GPU capability
```
docker run -it --rm \
    --gpus all \
    --shm-size=2g \
    -v "$(pwd)"/data/mysql_data:/var/lib/mysql \
    -v "$(pwd)/data":/data \
    -v "$(pwd)":/app \
    -p 3308:3306 \
    gen_ai_hiv
```
## Get the pretraining data
```
python3 getDataForPretrain.py --yaml="./PretrainSpecs.yml"
```

#### Random bits
Verify that the files do not have extraneous lines or double check where file references in files are located.
```
sudo grep -Rli "text" / 2>/dev/null
# Search for text in file and give line number
grep -n "text" my_script.py
```