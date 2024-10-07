from SelfiesDataHandler import SelfiesTokenizer
from transformers import BartForConditionalGeneration, BartConfig
import pandas as pd
from tokenizers import Tokenizer
from SelfiesDataHandler import SelfiesDataset
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import Trainer, TrainingArguments
import math
import yaml

# Load the tokenizer to get the vocab size

def train_and_save_BART(hyperparameters_dict, selfies_path="./data/selfies_subset.txt", bpe_path="./data/bpe/", save_to="./saved_model/"):
    TRAIN_BATCH_SIZE = hyperparameters_dict["TRAIN_BATCH_SIZE"]
    VALID_BATCH_SIZE = hyperparameters_dict["VALID_BATCH_SIZE"]
    TRAIN_EPOCHS = hyperparameters_dict["TRAIN_EPOCHS"]
    LEARNING_RATE = hyperparameters_dict["LEARNING_RATE"]
    WEIGHT_DECAY = hyperparameters_dict["WEIGHT_DECAY"]
    MAX_LEN = hyperparameters_dict["MAX_LEN"]


    # TODO: Figure out if this is which tokenizer or class utilized for this...
    tokenizer = Tokenizer.from_file(f"{bpe_path}/bpe.json")  # TODO: integrate this better #to just accept file path for this...

    # with open(args.hyperparameters_path) as file:
    #     hyperparameters = yaml.safe_load(file)
    #     for key in hyperparameters.keys():
    #         roberta_model.train_and_save_roberta_model(hyperparameters_dict=hyperparameters[key], selfies_path=args.prepared_data_path, robertatokenizer_path=args.roberta_fast_tokenizer_path, save_to="./saved_models/" + key + "_saved_model/")

    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),  # Set vocab size including special tokens
        max_position_embeddings=hyperparameters_dict["MAX_POSITION_EMBEDDINGS"],  # Adjust based on your needs
        encoder_layers=hyperparameters_dict["ENCODER_LAYERS"],
        decoder_layers=hyperparameters_dict["DECODER_LAYERS"],
        encoder_attention_heads=hyperparameters_dict["NUM_ENCODER_ATTENTION_HEADS"],
        decoder_attention_heads=hyperparameters_dict["NUM_DECODER_ATTENTION_HEADS"],
        encoder_ffn_dim=hyperparameters_dict["ENCODER_FFN_DIM"],
        decoder_ffn_dim=hyperparameters_dict["DECODER_FFN_DIM"],
        hidden_size=hyperparameters_dict["HIDDEN_SIZE"],  # Ensure this is divisible by the number of attention heads/ doesn't exist?
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=tokenizer.token_to_id("</s>"),
        mask_token_id=tokenizer.token_to_id("<mask>")  # Ensure this matches the ID used during pre-training
    )

    # Utilized in Trainer below
    def _model_init():
        return BartForConditionalGeneration(config=config)

    df = pd.read_csv(selfies_path, header=None) # SELFIES string path... this should be in .CSV format.
    

    # TODO: or is this the below one?
    tokenizer = SelfiesTokenizer(bpe_path) # TODO: GOT TO GET BPE SETUP RIGHT ON HERE...


    # # WIP with code from other...
    # pretrain_dataset = SelfiesDataset(csv_file='./ChEMBL34_druglike_activity.csv',
    #                                   tokenizer_path='./data/bpe_non/bpe.json', mode='finetune')
    # pretrain_loader = data_loader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    #
    # # Training setup
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # TODO: One of the possible exploration areas. Different loss functions?
    # criterion = torch.nn.CrossEntropyLoss()

    # TODO: Now need to get the pretrain and finetune stuffs on here and setup the optimizer and shizz....
    # criterion = torch.nn.CrossEntropyLoss() is set also... but where?
    # TRAIN_BATCH_SIZE = hyperparameters_dict["TRAIN_BATCH_SIZE"]
    # VALID_BATCH_SIZE = hyperparameters_dict["VALID_BATCH_SIZE"]
    # TRAIN_EPOCHS = hyperparameters_dict["TRAIN_EPOCHS"]
    # LEARNING_RATE = hyperparameters_dict["LEARNING_RATE"]
    # WEIGHT_DECAY = hyperparameters_dict["WEIGHT_DECAY"]
    # MAX_LEN = hyperparameters_dict["MAX_LEN"]
    # # VOCAB_SIZE: 800

    training_args = TrainingArguments(
        output_dir=save_to,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        save_total_limit=1,
        # disable_tqdm=True,
        # fp16=True
    )

    trainer = Trainer(
        model_init=_model_init,
        args=training_args,
        # data_collator=data_collator,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # prediction_loss_only=True,
    )

    print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
    trainer.train()
    print("training finished.")

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(save_to)