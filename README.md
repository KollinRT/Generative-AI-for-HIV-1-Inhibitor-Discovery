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


        new file:   ChEMBL34_druglike_activity_filtered_ringsless3_under550MW.csv
        new file:   ChEMBL34_selfies_non.csv
        new file:   Main5Finetuning_BaseNon.py
        new file:   WIP_BartSettings.py
        new file:   bart_model.py
        new file:   cleanedupExploreData.ipynb
        new file:   data/bpe_filter/bpe.json
        new file:   data/bpe_filter/merges.txt
        new file:   data/bpe_filter/vocab.json
        new file:   data_needed.sql
        new file:   environment.yml
        new file:   exploredata.ipynb
        new file:   models/WIP_BartSettings.py
        new file:   models/__init__.py
        new file:   models/create_hparam_set.py
        new file:   models/data/pretraining_hyperparameters.yml
        new file:   models/exampleArgs.txt
        new file:   models/hparams.yml
        new file:   models/hyperparameters.yml
        new file:   models/pretraining.yml
        new file:   outputs/selfies_filtered.txt
        new file:   train_for_pretrain.py

## How to configure the input files...