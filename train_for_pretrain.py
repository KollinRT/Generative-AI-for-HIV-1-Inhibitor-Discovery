# # import argparse
# #
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
# # parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
# # parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
# # parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
# # parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
# # # parser.add_argument("--roberta_fast_tokenizer_path", required=True, metavar="/path/to/robertafasttokenizer/", help="Directory of the RobertaTokenizerFast tokenizer. RobertaFastTokenizer only depends on the BPE Tokenizer and will be created regardless of whether it exists or not.")
# # parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
# # args = parser.parse_args()
# #
# # import pandas as pd
# #
# # # TODO: NEED TO apply the selfies conversion to build it... this would use the prepared_data_path, if I am looking right?
# #
# # try:
# #     df = pd.read_csv(args.selfies_dataset)
# # except FileNotFoundError:
# #     from prepare_dataset import prepare_dataset_for_pretrain
# #     print("No SEFLIES dataset")
# #     prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
# #     df = pd.read_csv(args.selfies_dataset)
# # print("We have a SELFIES set for ya!")
# #
# # print("Creating SELFIES.txt for tokenization.")
# # from os.path import isfile  # returns True if the file exists else False.
# #
# # if not isfile(args.prepared_data_path):
# #     from prepare_dataset import create_selfies_file
# #
# #     if args.subset_size != 0:
# #         create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
# #     else:
# #         create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
# # print("SELFIES .txt is ready for tokenization.")
# #
# # print("Creating BPE tokenizer.")
# # if not isfile(args.bpe_path + "/merges.txt"):
# #     import prepare_dataset
# #
# #     prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
# # print("BPE Tokenizer is ready.")
# # #
# # # print("Creating RobertaTokenizerFast.")
# # # if not isfile(args.roberta_fast_tokenizer_path + "/merges.txt"):
# # #     import roberta_tokenizer
# # #
# # #     roberta_tokenizer.save_roberta_tokenizer(path=args.bpe_path, save_to=args.roberta_fast_tokenizer_path)
# # # print("RobertaFastTokenizer is ready.")
# # #
# # import yaml
# # import WIP_BartSettings
# #
# # with open(args.hyperparameters_path) as file:
# #     hyperparameters = yaml.safe_load(file)
# #     for key in hyperparameters.keys():
# #         print("Starting pretraining with {} parameter set.".format(key))
# #         # TODO: INTEGRATE THIS MORE! Got to be able to call them to create the file...
# #         WIP_BartSettings.train_and_save_BART(hyperparameters_dict=hyperparameters[key],
# #                                                    selfies_path=args.prepared_data_path,
# #                                                    bpe_path=args.bpe_path,
# #                                                    save_to="./saved_models/" + key + "_saved_model/")
# #         print("Finished pretraining with {} parameter set.\n---------------\n".format(key))
# #
# # # Check to see
# # # def create_selfies_file(selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000, do_subset=True):
# # #     selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling
# # #
# # #     if do_subset:
# # #         selfies_subset = selfies_df.selfies[:subset_size]
# # #     else:
# # #         selfies_subset = selfies_df.selfies
# # #     selfies_subset = selfies_subset.to_frame()
# # #     selfies_subset["selfies"].to_csv(save_to, index=False, header=False)
# # #
# # # # create
#
# import argparse
# import pandas as pd
# import yaml
# import WIP_BartSettings
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
# parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
# parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
# parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
# parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
# parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
# args = parser.parse_args()
#
# try:
#     df = pd.read_csv(args.selfies_dataset)
# except FileNotFoundError:
#     from prepare_dataset import prepare_dataset_for_pretrain
#     print("No SEFLIES dataset")
#     prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
#     df = pd.read_csv(args.selfies_dataset)
# print("We have a SELFIES set for ya!")
#
# print("Creating SELFIES.txt for tokenization.")
# from os.path import isfile  # returns True if the file exists else False.
#
# if not isfile(args.prepared_data_path):
#     from prepare_dataset import create_selfies_file
#
#     if args.subset_size != 0:
#         create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
#     else:
#         create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
# print("SELFIES .txt is ready for tokenization.")
#
# print("Creating BPE tokenizer.")
# if not isfile(args.bpe_path + "/merges.txt"):
#     import prepare_dataset
#
#     prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
# print("BPE Tokenizer is ready.")
#
# with open(args.hyperparameters_path) as file:
#     hyperparameters = yaml.safe_load(file)
#
# print("Starting pretraining with HIDDEN_SIZE parameter set.")
# WIP_BartSettings.train_and_save_BART(
#     hyperparameters_dict=hyperparameters,
#     selfies_path=args.prepared_data_path,
#     bpe_path=args.bpe_path,
#     save_to="./saved_models/BART_saved_model/"
# )
# print("Finished pretraining with HIDDEN_SIZE parameter set.\n---------------\n")


### end

import argparse
import pandas as pd
import yaml
import WIP_BartSettings

parser = argparse.ArgumentParser()
parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/*.csv", help="Path of the SMILES dataset.")
parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/*.csv", help="Path of the SEFLIES dataset.")
parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
args = parser.parse_args()

try:
    df = pd.read_csv(args.selfies_dataset)
except FileNotFoundError:
    from prepare_dataset import prepare_dataset_for_pretrain
    print("No SEFLIES dataset")
    prepare_dataset_for_pretrain(path = args.smiles_dataset, save_to=args.selfies_dataset)
    df = pd.read_csv(args.selfies_dataset)
print("We have a SELFIES set for ya!")

# TODO: Need to apply the selfies translation somewhere here....

print("Creating SELFIES.txt for tokenization.")
from os.path import isfile  # returns True if the file exists else False.

if not isfile(args.prepared_data_path):
    from prepare_dataset import create_selfies_file

    if args.subset_size != 0:
        create_selfies_file(df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
    else:
        create_selfies_file(df, do_subset=False, save_to=args.prepared_data_path)
print("SELFIES .txt is ready for tokenization.")

print("Creating BPE tokenizer.")
if not isfile(args.bpe_path + "/merges.txt"):
    import prepare_dataset

    prepare_dataset.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
print("BPE Tokenizer is ready.")

with open(args.hyperparameters_path) as file:
    hyperparameters = yaml.safe_load(file)

print("Loaded hyperparameters:", hyperparameters)  # Debug print to check the loaded hyperparameters

# Access the 'BART' key in the hyperparameters dictionary
bart_hyperparameters = hyperparameters.get('BART', {})
print("Starting pretraining with HIDDEN_SIZE parameter set.")
WIP_BartSettings.train_and_save_BART(
    hyperparameters_dict=bart_hyperparameters,
    selfies_path=args.prepared_data_path,
    bpe_path=args.bpe_path,
    save_to="./saved_models/BART_saved_model/"
)
print("Finished pretraining with HIDDEN_SIZE parameter set.\n---------------\n")


