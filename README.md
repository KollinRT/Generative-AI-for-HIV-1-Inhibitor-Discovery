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
    database: 'chembl_34'
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
python3 MakeDefaultHyperparams.py. --file "./PretrainSpecs.yml"
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
