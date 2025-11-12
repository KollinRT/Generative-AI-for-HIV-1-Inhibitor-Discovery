# How to run
This is how

python3 train_for_pretrain.py --smiles_dataset ""


python3 train_for_pretrain.py --smile_dataset="ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv" --selfies_dataset="ChEMBL34_selfies_non.csv" --bpe_path="./data/bpe_filter/"

python3 train_for_pretrain.py --smiles_dataset="./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv" --selfies_dataset="./ChEMBL34_selfies_non.csv" --prepared_data_path="./outputs/" --bpe_path="./data/bpe_filter/"

--prepared_data_path="./outputs/prepared/"

For non-filtered sample dataset...
`python3 train_for_pretrain.py --smiles_dataset="./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv" --selfies_dataset="./ChEMBL34_selfies_non.csv" --prepared_data_path="./outputs/selfies_filtered.txt" --bpe_path="./data/bpe_filter/"`
is an example run of it...

To run

python3 train_for_pretrain.py --smiles_dataset="./ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv" --selfies_dataset="./ChEMBL34_selfies_non.csv" --prepared_data_path="./outputs/selfies_filtered.txt" --bpe_path="./data/bpe_filter/" --hyperparameters_path="BARTModel_WIP_pretraining.yml"

need to use `pip install accelerate -U` to get it working with training 

## How to configure the input files...

In order for set up to run, we need to first download data for [ChEMBL_34](https://chembl.gitbook.io/chembl-interface-documentation/downloads), which we would download the ChEMBLdb and the database format of your choice.

To configure the `FinetuneSpecs.yml` file, we need to figure out two things.

```yaml
pymysql_info:
    host: 'host'
    user: 'user'
    password: 'pass'
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
Which we would pass into the file...
To configure the `BARTModel_WIP_pretraining.yml` file we need to construct a yaml file with the following parameters... 
These are the following parameters defaultly set in a similar paper (SEE THIS... molbart or selformer or idk... different usage...)

```yaml
BART:
  HIDDEN_SIZE: 768
  TRAIN_BATCH_SIZE: 16
  VALID_BATCH_SIZE: 8
  TRAIN_EPOCHS: 100
  LEARNING_RATE: 0.00005
  WEIGHT_DECAY: 0.01
  MAX_LEN: 128
  ENCODER_LAYERS: 12
  DECODER_LAYERS: 12
  NUM_ENCODER_ATTENTION_HEADS: 12
  NUM_DECODER_ATTENTION_HEADS: 12
  ENCODER_FFN_DIM: 3072
  DECODER_FFN_DIM: 3072
  VOCAB_SIZE: 30000
  MAX_POSITION_EMBEDDINGS: 514
  NUM_ATTENTION_HEADS: 12
  NUM_HIDDEN_LAYERS: 8
  TYPE_VOCAB_SIZE: 1
```
<!-- TODO: NEW `VOCAB_SIZE` may have to be adapted from actual code... not just defaultly set. Would have to think about what VOCAB_SIZE to use....  -->
<!-- TODO: NEW Am I using BPE still? I think I am? I think I need around 50 to 500 tokens or 1000 to 10000 but probably 1000 at most? Not sure. Explore vocab sizes by querying it.... -->

<!-- SOOOO, this uses a different model setup then my old portion? Or I am just using already established bpe_vocab with the arg.bpe_path... -->

```bash
python3 getDataForFinetune.py --yaml "./FinetuneSpecs.yml"
```

This returns a `.csv` file with 
`canonical_smiles,MW,numC,chain_length,cLogP,numRings,IC50,site_name`...
Next we proceed with 

```bash
python3 MakeDefaultHyperparams.py --file "./PretrainSpecs.yml"
```
This will make the `combined_config.yml` file with some default hyperparameters...

Then this makes us our `model_name_{key}` .csv of the data with the columns needed for pretraining...

To proceed with finetuning, we need to....

To run this pretraining we use
```python3
python3 train_for_pretrain.py --yaml="./PretrainSpecs.yml"
```

```python3
python3 train_for_pretrain.py --hyperparameters_path="./PretrainSpecs.yml"
```

```bash
python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml"
```

Going to have to add logging information for loss to create a graph for each epoch or every 5 epoch logged into a .csv file that I can graph at the end...
- - Maybe even have it create the graphs at the end of each go around and make it be pretrain and even save the config info for each model... that way it allows for easy replicability!


## In training loop
Need to add epoch % 5 == 0, log info on loss and epoch no...


### Logical execution
```bash
conda activate thesisproj
```
#### 1. Get the data
utilize the file
`getDataForPretrain.yml` 
```bash
python3 getDataForPretrain.py --yaml="./PretrainSpecs.yml"
```
gets you going to generate the files. This can then be fed into the next step.

[//]: # (##### Need to make the default parameters)

[//]: # (```bash)

[//]: # (python3 MakeDefaultHyperparams.py --file="./PretrainSpecs.yml")

[//]: # (```)

#### 2. Generate the combined_config file
utilize the file
`train_for_pretrain.py`

[//]: # (```bash)

[//]: # (python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml")

[//]: # (```)

```bash
python3 MakeDefaultHyperparams.py --file="./PretrainSpecs.yml"
```
[//]: # (> Combined configuration saved to combined_config.yml)

But first you have to generate the data for the `./combined_config.yml` file.
This is in the 
`MakeDefaultHyperparams.py` 
file along with the default hyperparams config.
Run it then generate the `./combined_config.yml`

#### 3a. Generate the fingerprints and the clusters.
The fingerprints for the models can be generated with the
`generateClusters.py` and `generateFingerprints.py`
files. These may be incorporated into the `train_for_pretrain.py` file, but still would have to get it working, as it is not currently but works independently...  
**TODO**:
- [ ] Get the standalone files to work well and consistently and document the process.
    - [ ] Get this implemented into the current logic and not standalone?

This will perform the clustering that will interplay with the `ClusteredSelfiesDataset` class in `train_for_pretrain.py`.
This will add the fingerprints and cluster columns into the df (csv) file that will be utilized to help select the most likely singleton drugs for use in validation splitting.
- should be one-offs, I hope? The last clusterIDs are lower in total count?


Should be the following basics
```bash
python3 prepro_ClustFing.py 
```
At the moment to generate selfies, generate fingerprints, and then generate clusters...



#### 3b. Train the pre-train model
utilize the file
`train_for_pretrain.py`
This will work with the `./combined_config.yml` file since it needs the hyperparameters.  


```bash
python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml"
```
This will train the model with the SELFIES text.


##### Tech Specs
Intel Core i9-13900KF Processor (24 core (8P/16E)/32 Threads)  
128 GB RAM 4800 MHz  
RTX 4090 24GB VRAM  




#### TODO:   
- [ ] Get `getDataForFinetune.py` working for the finetuning dataset.
    - [ ] Need to get random sampling done for 9:1 split for finetuning.
- [ ] explore hyperparameter optimization
    - [ ] this could include a pytorch LRScheduler... 
    - [ ] this could also include trying adagrad? Maybe optimizing hyperparameter dimensions?
      - [ ] check the post more...

- [ ] This could be trying `"ReduceLROnPlateau" class (https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py), would this work well?


#### Definitive Workflow make into a bash script
```bash
conda activate thesisproj

# Round 1
python getDataForPretrain.py --parallel --yaml "./PretrainSpecs.yml"
## If not-parallel
python getDataForPretrain.py --yaml "./PretrainSpecs.yml"

# To generate `combined_config.yml`
python3 MakeDefaultHyperparams.py --file="./PretrainSpecs.yml"

# Convert to SELFIES
python addSelfiesDescription.py --smiles_column="canonical_smiles" --hyperparameters_path="./combined_config.yml"

python3 prepro_ClustFing.py 

python3 train_for_pretrain.py --hyperparameters_path="./combined_config.yml"



# Round 2/3
python3 getDataForPretrain.py --yaml="./PretrainSpecs.yml" --round="2"


# Round 3
python3 train_for_pretrain.py --hyperparameters_path="./combined_config_steps.yml"


# Finetuning
python3 train_for_finetune.py --hyperparameters_path="./combined_config_WIP_FT.yml"

```


Obtain data for Rounds 2 and 3:
```bash
bash ExploreThesis/WIP_Thesis/scripts/download_zinc.sh
```
Which will download all the ZINC15 druglike molecules. Then to get the 10M from ZINC15 we need to
- Put all mols into one `.smi` file
- Sample 10M molecules from the ZINC15 dataset.

Then to work to process the 860M ZINC15 druglike molecules:
`process_in_chunk.py` is utilized to make 860 1M molecules file to then process into a parquet file. This parquet file will
interact with Dask in order to utilize the big data streaming for Round 3.

# Steps

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

### Download Druglike Molecules from ZINC 15
[ZINC15 Tranches](https://zinc15.docking.org/tranches/home/)

Click the 3x3 dots button <img src="./images/ZINC15_dots.png" width="25" height="25"> and click the Druglike filter and click off of 5 on left and 500 on top. This will get the data in download that can be downloaded with a script via WGET. Run that script in the data directory to generate the data.

or use the existing [File in repo](./scripts/download_zinc.sh)
```bash
bash download_zinc.sh
```
And let the whole file directory download. Much like the ChEMBL DB population, this might take several hours so please set time aside to accommodate this.

<!-- ![3x3 Dots on ZINC15 Tranches Page](/images/ZINC15_dots.png "3x3 Tranch Dots") -->


#### Get Round 2 Data 10M
Once the data is in the directory needed we can run the script `combine_zinc15.sh` which will combine all the ZINC15 `*.smi` files so that we can then sample 10M for Round 2.

```
find . -type f -name "*.smi" -exec cat {} + > zinc15_all_raw.smi
or
awk 'FNR==1 && NR!=1 {next} {print}' $(find . -type f -name "*.smi") > zinc15_all_raw.smi
```
The top one is quicker but is also grabbing the header multiple times. The bottom one times longer but is not.
Then one could run 

`WIP_Thesis/scripts/Round2/train_word_level_tokenizer_round2.py` is the file used to create the tokenizer for Round 2. Just need to now find the file that SELFIES the other files... Process the ChEMBL 35 datas...
```
shuf -n 10000000 zinc15_all_raw.smi > zinc15_sampled_10M.smi
```
to get 10M molecules needed for Round 2. This gives me the molecules with `smiles` and `zinc_id`. Need to check.

Then from here we can combine the 10M sampled with the 2.3M pretrained molecules to then do FPs.

#### Get Round 3 Data ~860M
This is a tricky round...!

## Build the Docker image
```
docker build -t uv-run2-new23 .
```
## Run the Docker container
```
docker run -it --rm \
	--shm-size=2g \
    -v "$(pwd)"/data/mysql_data:/var/lib/mysql \
    -v "$(pwd)/data":/data \
    -v "$(pwd)":/app \
    -p 3307:3306 \
    uv-run2-new23
```

## Get the pretraining data
```
python3 getDataForPretrain.py --yaml="./PretrainSpecs.yml"
```



# TODO
- [ ] Get the thing working. Get the 10M for round 2 with fingerprints working. Figure out what is needed to do this!
- [ ] Once this is done we can then run the basic code...
- [ ] `model_base_selfies_only` is where I erase the top header for the tokenization portion....
- [ ] Make sure that I have clearly delineated scripts for Round 1 and Round 2 versus Round 3 which is global steps.

- [ ] Make `train_for_pretrain.py` flexible between epochs and steps.

df1 = pd.read_csv("model_base_selfies_only.csv")
get process selfies script

```
sudo grep -Rli "text" / 2>/dev/null
# Search for text in file and give line number
grep -n "text" my_script.py
```
```
python process_10M_parallel.py
```