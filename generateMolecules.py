"""
This script is for generating molecules from the pretrained and fine-tuned model for use for evaluating model performance.

#TODO:
1. Load the finetuned model
2. Generate molecules
3. Save the generated molecules to a file
4. Evaluate the generated molecules
5. Save the evaluation results to a file
6. Plot the evaluation results?
7. Save the plot?
8. Compare to the baseline model
9. Save the comparison results to a file
10. Plot the comparison results?
11. Save the plot?

This will be shittily the steps required...
CLEANUP!
"""

import torch
from debugpy.launcher import output
from tokenizers import Tokenizer
from transformers import BartForConditionalGeneration


# def load_model(model_path, device='cuda'):
#     """Load the pre-trained BART model."""
#     model = torch.load(model_path)
#     model = model.to(device)
#     model.eval()  # Set model to evaluation mode
#     return model
# def load_model(model_path, device='cuda'):
#     """Load the pre-trained BART model."""
#     model = BartForConditionalGeneration.from_pretrained(model_path)
#     model = model.to(device)
#     model.eval()  # Set model to evaluation mode
#     return model
def load_model(model_path, device='cuda'):
    """Load the pre-trained BART model."""
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  # Initialize the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model



def load_tokenizer(tokenizer_path):
    """Load the tokenizer used for both encoding and decoding."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

# def load_tokenizer(tokenizer_path):
#     """Load the tokenizer used for both encoding and decoding."""
#     tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
#     return tokenizer

# def load_tokenizer(tokenizer_path):
#     """Load the tokenizer used for both encoding and decoding."""
#     tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
#     return tokenizer


# def generate_text(model,
#                   tokenizer,
#                   input_text,
#                   max_length=50,
#                   num_return_sequences=1,
#                   num_beams=4):
#     """Generate text using the model and tokenizer."""
#     # Encode the input text using the custom tokenizer
#     # encoded_input = tokenizer.encode(input_text)
#     # input_ids = torch.tensor([encoded_input.ids
#     #                           ])  # Convert to a tensor manually
#     input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

#     # Ensure model is in evaluation mode and tensors are moved to the same device as the model
#     model.eval()
#     input_ids = input_ids.to(next(model.parameters()).device)

#     # Generate predictions using the model
#     # TODO: Play around with the parameters to see how they affect the generated molecules
#     # I.e. early_stopping can be toggled on and off, num_return_sequences is definitely set!
#     # Figure out how to do continual and not batched generation
#     outputs = model.generate(input_ids,
#                              max_length=max_length,
#                              num_return_sequences=num_return_sequences,
#                              num_beams=num_beams,
#                              early_stopping=True,
#                              no_repeat_ngram_size=2)

#     # Decode and return each generated sequence
#     generated_texts = [
#         tokenizer.decode(output, skip_special_tokens=True)
#         for output in outputs.cpu().numpy()
#     ]
#     return generated_texts

# def generate_text(model, tokenizer, input_text, max_length=50, num_return_sequences=1, num_beams=4):
#     """Generate molecules using the model and tokenizer."""
#     # input_ids = tokenizer.encode(input_text, return_tensors='pt').ids
#     input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Returns a tensor
#     input_ids = torch.tensor([input_ids]).to(model.device)
#
#     outputs = model.generate(input_ids,
#                              max_length=max_length,
#                              num_return_sequences=num_return_sequences,
#                              num_beams=num_beams,
#                              early_stopping=True,
#                              no_repeat_ngram_size=2)
#
#     # Decode and return each generated sequence
#     generated_texts = [
#         tokenizer.decode(output.tolist(), skip_special_tokens=True)
#         for output in outputs
#     ]
#     return generated_texts

def generate_text(model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5):
    model.eval()  # Set the model to evaluation mode

    # ✅ Encode input text correctly
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    # ✅ Generate text
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )

    # ✅ Decode generated text correctly
    decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return decoded_texts


def benchmark_generated_molecules():
    """
    Generate molecules using the model and tokenizer.
    This should include novelty and uniqueness
    Utilize the training set...
    - _{key} will be utilized from ./data/ folder
    - how are we marking train vs validation? This should be the same...
    - Get the code here...

    formula for novelty:
        - here
    formula for uniqueness:
        - here

    """
    pass

# TODO: NEW include sampling...

# if __name__ == "__main__":
#     # Specify the paths to the model and tokenizer
#     model_path = './selfies_BART_druglike_fine_tuned.pth'
#     tokenizer_path = './data/bpe/bpe.json'

#     # Set the device to use (either 'cuda' or 'cpu')
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load the model and tokenizer
#     model = load_model(model_path, device)
#     tokenizer = load_tokenizer(tokenizer_path)

#     # Input text for generation
#     input_text = "[C][S][Br]"

#     # Generate text
#     generated_texts = generate_text(model,
#                                     tokenizer,
#                                     input_text,
#                                     max_length=1000,
#                                     num_return_sequences=2000,
#                                     num_beams=2000)

#     # Print generated texts
#     for text in generated_texts:
#         print("Generated Text:", text)

# if __name__ == "__main__":
#     # Specify the paths to the model and tokenizer
#     model_path = './selfies_BART_druglike_fine_tuned'  # Note: No .pth extension for from_pretrained
#     tokenizer_path = 'facebook/bart-base'  # Use the pre-trained tokenizer from Hugging Face

#     # Set the device to use (either 'cuda' or 'cpu')
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load the model and tokenizer
#     model = load_model(model_path, device)
#     tokenizer = load_tokenizer(tokenizer_path)

#     # Input text for generation
#     input_text = "[C][S][Br]"

#     # Generate text
#     generated_texts = generate_text(model,
#                                     tokenizer,
#                                     input_text,
#                                     max_length=1000,
#                                     num_return_sequences=2000,
#                                     num_beams=2000)

#     # Print generated texts
#     for text in generated_texts:
#         print("Generated Text:", text)

from tokenizers import Tokenizer
import os

def load_tokenizer(tokenizer_path):
    # ✅ Check if tokenizer_path is a directory
    if os.path.isdir(tokenizer_path):
        # Look for a tokenizer JSON file inside
        possible_files = ["bpe.json", "tokenizer.json", "vocab.json"]
        for file in possible_files:
            full_path = os.path.join(tokenizer_path, file)
            if os.path.exists(full_path):
                tokenizer_path = full_path
                break
        else:
            raise FileNotFoundError(f"❌ No valid tokenizer file found in {tokenizer_path}")

    # ✅ Load the tokenizer from the correct file
    return Tokenizer.from_file(tokenizer_path)


if __name__ == "__main__":
    # # Specify the paths to the model and tokenizer
    # model_path = f'/home/kollin/Desktop/ThesisBU/WIP_Thesis/selfies_BART_finetuned_model_nada.pth'  # Path to the saved state dictionary
    # # tokenizer_path = 'facebook/bart-base'  # Use the pre-trained tokenizer from Hugging Face
    # # # TODO: New, update the tokenizer_path above!
    # """
    # #     tokenizer_path = './data/bpe/bpe.json'
    # Something like that above.... data _{key} yeah...
    # """
    # tokenizer_path = './home/kollin/Desktop/ThesisBU/WIP_Thesis/data/bpe_filter_model_nada'
    #
    # # Set the device to use (either 'cuda' or 'cpu')
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # # Load the model and tokenizer
    # model = load_model(model_path, device)
    # tokenizer = load_tokenizer(tokenizer_path)
    #
    # # Input text for generation
    # input_text = "[C][S][Br]" # Is this utilized? I do not see it!
    #
    # # Generate text
    # generated_texts = generate_text(model,
    #                                 tokenizer,
    #                                 input_text,
    #                                 max_length=1000,
    #                                 num_return_sequences=2000,
    #                                 num_beams=2000)
    #
    # # Print generated texts
    # for text in generated_texts:
    #     print("Generated Text:", text)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    #
    # # ✅ Set paths
    # model_path = f'./selfies_BART_finetuned_model_nada.pth'  # Update with the correct model key
    # base_model_name = "facebook/bart-base"  # Update if a different base model was used
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load model configuration (modify as needed based on your training config)
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # Adjust configuration to match fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches the fine-tuned model
    # config.max_position_embeddings = 516  # Ensure position embeddings match
    # config.encoder_layers = 12  # Update with actual layers used
    # config.decoder_layers = 12  # Update with actual layers used
    #
    # # ✅ Load model architecture with correct configuration
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Load fine-tuned weights (allow mismatches)
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # Handle unexpected/missing keys
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # input_text = "[C][S][Br]"  # Example input
    # tokenizer = load_tokenizer("./data/bpe_filter_model_nada")  # Ensure correct tokenizer path
    #
    # generated_texts = generate_text(
    #     model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5
    # )
    #
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    #
    # # ✅ Set paths
    # model_path = f'./selfies_BART_finetuned_model_nada.pth'  # Update with the correct model key
    # base_model_name = "facebook/bart-base"  # Change if you used a different base model
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load base model configuration
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # ✅ Override configuration to match fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches fine-tuned model
    # config.max_position_embeddings = 516  # Set to 516 to match your fine-tuned model
    #
    # # ✅ Load model architecture with updated configuration
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Resize positional embeddings BEFORE loading state_dict
    # model.model.encoder.embed_positions.weight = torch.nn.Parameter(
    #     torch.zeros((516, config.d_model))  # Match the 516 size from fine-tuned model
    # )
    # model.model.decoder.embed_positions.weight = torch.nn.Parameter(
    #     torch.zeros((516, config.d_model))  # Match the 516 size from fine-tuned model
    # )
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Load state_dict with strict=False to ignore mismatches
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # input_text = "[C][S][Br]"
    # tokenizer_path = "./data/bpe_filter_model_nada/bpe.json"  # Ensure this is the actual file
    # tokenizer = load_tokenizer(tokenizer_path)
    # generated_texts = generate_text(model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5)
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    # from tokenizers import Tokenizer
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # tokenizer_path = "./data/bpe_filter_model_nada/bpe.json"  # Ensure correct file path
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load model configuration
    # config = BartConfig.from_pretrained("facebook/bart-base")
    # config.vocab_size = 360  # Adjust to fine-tuned model
    # config.max_position_embeddings = 516  # Ensure matching size
    #
    # # ✅ Load model with updated configuration
    # model = BartForConditionalGeneration(config)
    #
    # print(model.model.encoder.embed_positions.weight.shape)
    # print(model.model.decoder.embed_positions.weight.shape)
    #
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict, strict=False)
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Load tokenizer with improved error handling
    # tokenizer = load_tokenizer(tokenizer_path)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # base_model_name = "facebook/bart-base"  # Adjust if another model was used as the base
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load base model configuration
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # ✅ Adjust configuration to match the fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches fine-tuned model
    # config.max_position_embeddings = 516  # Ensure correct embedding size from fine-tuned model
    #
    # # ✅ Load model architecture with updated configuration
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Resize positional embeddings BEFORE loading state_dict
    # # Fix mismatch between 518 vs. 516
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = torch.nn.Embedding(516, config.d_model)
    #     model.model.decoder.embed_positions = torch.nn.Embedding(516, config.d_model)
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Fix potential shape mismatch by trimming or expanding embeddings
    # with torch.no_grad():
    #     # Load embeddings from checkpoint
    #     checkpoint_encoder_embed = state_dict["model.encoder.embed_positions.weight"]
    #     checkpoint_decoder_embed = state_dict["model.decoder.embed_positions.weight"]
    #
    #     # Resize to 518 if needed
    #     if checkpoint_encoder_embed.shape[0] < 518:
    #         pad_size = (518 - checkpoint_encoder_embed.shape[0], config.d_model)
    #         checkpoint_encoder_embed = torch.cat([checkpoint_encoder_embed, torch.zeros(pad_size)], dim=0)
    #         checkpoint_decoder_embed = torch.cat([checkpoint_decoder_embed, torch.zeros(pad_size)], dim=0)
    #
    #     # Assign adjusted embeddings back into state_dict
    #     state_dict["model.encoder.embed_positions.weight"] = checkpoint_encoder_embed
    #     state_dict["model.decoder.embed_positions.weight"] = checkpoint_decoder_embed
    #
    # # ✅ Load state_dict with strict=False to ignore minor mismatches
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"
    # generated_texts = generate_text(model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5)
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # base_model_name = "facebook/bart-base"  # Adjust if another model was used as the base
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load base model configuration
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # ✅ Adjust configuration to match the fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches fine-tuned model
    # config.max_position_embeddings = 518  # Change this to match your trained model!
    #
    # # ✅ Load model architecture with updated configuration
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Move model to device before modifying embeddings
    # model.to(device)
    #
    # # ✅ Resize positional embeddings BEFORE loading state_dict
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model).to(
    #         device)
    #     model.model.decoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model).to(
    #         device)
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Fix potential shape mismatch by trimming or expanding embeddings
    # with torch.no_grad():
    #     # Move checkpoint embeddings to the same device as the model
    #     checkpoint_encoder_embed = state_dict["model.encoder.embed_positions.weight"].to(device)
    #     checkpoint_decoder_embed = state_dict["model.decoder.embed_positions.weight"].to(device)
    #
    #     # Resize to 518 if needed
    #     if checkpoint_encoder_embed.shape[0] < config.max_position_embeddings:
    #         pad_size = (config.max_position_embeddings - checkpoint_encoder_embed.shape[0], config.d_model)
    #         checkpoint_encoder_embed = torch.cat([checkpoint_encoder_embed, torch.zeros(pad_size, device=device)],
    #                                              dim=0)
    #         checkpoint_decoder_embed = torch.cat([checkpoint_decoder_embed, torch.zeros(pad_size, device=device)],
    #                                              dim=0)
    #
    #     # Assign adjusted embeddings back into state_dict
    #     state_dict["model.encoder.embed_positions.weight"] = checkpoint_encoder_embed
    #     state_dict["model.decoder.embed_positions.weight"] = checkpoint_decoder_embed
    #
    # # ✅ Load state_dict with strict=False to ignore minor mismatches
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device (ensuring all tensors are on the same device)
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"
    # generated_texts = generate_text(model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5)
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # tokenizer_path = "./data/bpe_filter_model_nada"  # Update if necessary
    # base_model_name = "facebook/bart-base"
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load base model configuration
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # ✅ Adjust configuration to match the fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches fine-tuned model
    # config.max_position_embeddings = 518  # Change this to match your trained model!
    #
    # # ✅ Load model architecture with updated configuration
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Move model to device before modifying embeddings
    # model.to(device)
    #
    # # ✅ Resize positional embeddings BEFORE loading state_dict
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model).to(
    #         device)
    #     model.model.decoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model).to(
    #         device)
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Fix potential shape mismatch by trimming or expanding embeddings
    # with torch.no_grad():
    #     # Move checkpoint embeddings to the same device as the model
    #     checkpoint_encoder_embed = state_dict["model.encoder.embed_positions.weight"].to(device)
    #     checkpoint_decoder_embed = state_dict["model.decoder.embed_positions.weight"].to(device)
    #
    #     # Resize to 518 if needed
    #     if checkpoint_encoder_embed.shape[0] < config.max_position_embeddings:
    #         pad_size = (config.max_position_embeddings - checkpoint_encoder_embed.shape[0], config.d_model)
    #         checkpoint_encoder_embed = torch.cat([checkpoint_encoder_embed, torch.zeros(pad_size, device=device)],
    #                                              dim=0)
    #         checkpoint_decoder_embed = torch.cat([checkpoint_decoder_embed, torch.zeros(pad_size, device=device)],
    #                                              dim=0)
    #
    #     # Assign adjusted embeddings back into state_dict
    #     state_dict["model.encoder.embed_positions.weight"] = checkpoint_encoder_embed
    #     state_dict["model.decoder.embed_positions.weight"] = checkpoint_decoder_embed
    #
    # # ✅ Load state_dict with strict=False to ignore minor mismatches
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device (ensuring all tensors are on the same device)
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Load tokenizer
    # try:
    #     tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    #     print(f"✅ Tokenizer loaded successfully from {tokenizer_path}")
    # except Exception as e:
    #     print(f"❌ Error loading tokenizer: {e}")
    #     exit()

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # base_model_name = "facebook/bart-base"
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load base model configuration
    # config = BartConfig.from_pretrained(base_model_name)
    #
    # # ✅ Adjust configuration to match the fine-tuned model
    # config.vocab_size = 360  # Ensure vocab size matches fine-tuned model
    # config.max_position_embeddings = 516  # Ensure correct embedding size from fine-tuned model
    #
    # # ✅ Load base model with updated configuration
    # model = BartForConditionalGeneration.from_pretrained(base_model_name, config=config)
    #
    # # ✅ Load fine-tuned state_dict
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Resize learned positional embeddings properly
    # model.model.encoder.embed_positions = model.model.encoder.embed_positions.to(device)
    # model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    #
    # # ✅ Load state dict without breaking positional embeddings
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"
    # generated_texts = generate_text(model, tokenizer, input_text, max_length=100, num_return_sequences=5, num_beams=5)
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # tokenizer_path = "./data/bpe_filter_model_nada"  # Ensure correct tokenizer path
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load the **fine-tuned** model's config instead of base bart-base
    # config = BartConfig(
    #     vocab_size=360,  # SELFIES vocab size
    #     max_position_embeddings=518,  # Updated positional embeddings
    #     d_model=768,  # Same as bart-base
    #     encoder_layers=12,
    #     decoder_layers=12,
    #     encoder_attention_heads=12,
    #     decoder_attention_heads=12,
    #     encoder_ffn_dim=3072,
    #     decoder_ffn_dim=3072,
    #     activation_function="gelu",
    #     pad_token_id=1,  # Ensure this matches your training setup
    #     eos_token_id=2,  # Ensure this matches your training setup
    #     bos_token_id=0,  # Ensure this matches your training setup
    # )
    #
    # # ✅ Load model with **empty weights** (matching fine-tuned structure)
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Resize token embeddings before loading weights
    # model.resize_token_embeddings(config.vocab_size)
    #
    # # ✅ Adjust positional embeddings before loading weights
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model)
    #     model.model.decoder.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.d_model)
    #
    # # ✅ Load state_dict, ignoring mismatched sizes
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to GPU if available
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Load the tokenizer (ensure this tokenizer matches the fine-tuned vocabulary)
    # tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    # print(f"✅ Tokenizer loaded successfully from {tokenizer_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"  # Example SELFIES input
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    #
    # generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=5, num_beams=5)
    # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # tokenizer_path = "./data/bpe_filter_model_nada"  # Ensure correct tokenizer path
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load the **fine-tuned** model's config instead of bart-base
    # config = BartConfig(
    #     vocab_size=360,  # SELFIES vocab size
    #     max_position_embeddings=516,  # ✅ Ensure this matches fine-tuned model (NOT 518)
    #     d_model=768,  # Same as bart-base
    #     encoder_layers=12,
    #     decoder_layers=12,
    #     encoder_attention_heads=12,
    #     decoder_attention_heads=12,
    #     encoder_ffn_dim=3072,
    #     decoder_ffn_dim=3072,
    #     activation_function="gelu",
    #     pad_token_id=1,  # Ensure this matches your training setup
    #     eos_token_id=2,  # Ensure this matches your training setup
    #     bos_token_id=0,  # Ensure this matches your training setup
    # )
    #
    # # ✅ Load model with **empty weights** (matching fine-tuned structure)
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Load fine-tuned weights
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Resize token embeddings before loading weights
    # model.resize_token_embeddings(config.vocab_size)
    #
    # # ✅ Adjust positional embeddings **to 516 BEFORE loading weights**
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = torch.nn.Embedding(516, config.d_model)
    #     model.model.decoder.embed_positions = torch.nn.Embedding(516, config.d_model)
    #
    # # ✅ Ensure state_dict embeddings match `516`
    # if "model.encoder.embed_positions.weight" in state_dict:
    #     state_dict["model.encoder.embed_positions.weight"] = state_dict["model.encoder.embed_positions.weight"][:516, :]
    # if "model.decoder.embed_positions.weight" in state_dict:
    #     state_dict["model.decoder.embed_positions.weight"] = state_dict["model.decoder.embed_positions.weight"][:516, :]
    #
    # # ✅ Load state_dict, ignoring mismatched sizes
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to GPU if available
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Load the tokenizer (ensure this tokenizer matches the fine-tuned vocabulary)
    # tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    # print(f"✅ Tokenizer loaded successfully from {tokenizer_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"  # Example SELFIES input
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    #
    # generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=5, num_beams=5)
    # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #
    # print("Generated Texts:", generated_texts)

    # import torch
    # from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    # from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
    #
    # # ✅ Set paths
    # model_path = "./selfies_BART_finetuned_model_nada.pth"
    # tokenizer_path = "./data/bpe_filter_model_nada"  # Ensure correct tokenizer path
    #
    # # ✅ Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # ✅ Load model config (ensure it matches fine-tuned model)
    # config = BartConfig(
    #     vocab_size=360,  # SELFIES vocab size
    #     max_position_embeddings=514,  # ✅ Ensure this matches fine-tuned model (NOT 518)
    #     d_model=768,
    #     encoder_layers=12,
    #     decoder_layers=12,
    #     encoder_attention_heads=12,
    #     decoder_attention_heads=12,
    #     encoder_ffn_dim=3072,
    #     decoder_ffn_dim=3072,
    #     activation_function="gelu",
    #     pad_token_id=1,
    #     eos_token_id=2,
    #     bos_token_id=0,
    # )
    #
    # # ✅ Initialize model with **correct architecture**
    # model = BartForConditionalGeneration(config)
    #
    # # ✅ Fix positional embeddings using `BartLearnedPositionalEmbedding`
    # with torch.no_grad():
    #     model.model.encoder.embed_positions = BartLearnedPositionalEmbedding(516, config.d_model)
    #     model.model.decoder.embed_positions = BartLearnedPositionalEmbedding(516, config.d_model)
    #
    # # ✅ Load fine-tuned state_dict
    # state_dict = torch.load(model_path, map_location=device)
    #
    # # ✅ Ensure embeddings are resized properly before loading state_dict
    # model.resize_token_embeddings(config.vocab_size)
    #
    # # ✅ Fix potential size mismatches for positional embeddings
    # if "model.encoder.embed_positions.weight" in state_dict:
    #     state_dict["model.encoder.embed_positions.weight"] = state_dict["model.encoder.embed_positions.weight"][:516, :]
    # if "model.decoder.embed_positions.weight" in state_dict:
    #     state_dict["model.decoder.embed_positions.weight"] = state_dict["model.decoder.embed_positions.weight"][:516, :]
    #
    # # ✅ Load the state_dict while ignoring size mismatches
    # model.load_state_dict(state_dict, strict=False)
    #
    # # ✅ Move model to device
    # model.to(device)
    #
    # print(f"✅ Fine-tuned model loaded successfully from {model_path}")
    #
    # # ✅ Load tokenizer
    # tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    # print(f"✅ Tokenizer loaded successfully from {tokenizer_path}")
    #
    # # ✅ Generate text
    # input_text = "[C][S][Br]"  # Example SELFIES input
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    #
    # generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=5, num_beams=5)
    # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #
    # print("Generated Texts:", generated_texts)

    import torch
    from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
    from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding

    # ✅ Set paths
    model_path = "./selfies_BART_finetuned_model_nada.pth"
    tokenizer_path = "./data/bpe_filter_model_nada"  # Ensure correct tokenizer path

    # ✅ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load model config (ensure it matches fine-tuned model)
    config = BartConfig(
        vocab_size=360,  # SELFIES vocab size
        max_position_embeddings=514,  # ✅ Fix positional embedding size
        d_model=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        activation_function="gelu",
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=0,
    )

    # ✅ Initialize model with the correct architecture
    model = BartForConditionalGeneration(config)

    # ✅ Fix positional embeddings using `BartLearnedPositionalEmbedding`
    with torch.no_grad():
        model.model.encoder.embed_positions = BartLearnedPositionalEmbedding(516, config.d_model)
        model.model.decoder.embed_positions = BartLearnedPositionalEmbedding(516, config.d_model)

    # ✅ Load fine-tuned state_dict
    state_dict = torch.load(model_path, map_location=device)

    # ✅ Trim/Expand `embed_positions.weight` to match `514`
    if "model.encoder.embed_positions.weight" in state_dict:
        state_dict["model.encoder.embed_positions.weight"] = torch.nn.functional.pad(
            state_dict["model.encoder.embed_positions.weight"][:518, :],
            (0, 0, 0, max(0, 518 - state_dict["model.encoder.embed_positions.weight"].shape[0]))
        )
    if "model.decoder.embed_positions.weight" in state_dict:
        state_dict["model.decoder.embed_positions.weight"] = torch.nn.functional.pad(
            state_dict["model.decoder.embed_positions.weight"][:518, :],
            (0, 0, 0, max(0, 518 - state_dict["model.decoder.embed_positions.weight"].shape[0]))
        )

    # ✅ Resize embeddings before loading state_dict
    model.resize_token_embeddings(config.vocab_size)

    # ✅ Load state_dict
    model.load_state_dict(state_dict, strict=False)

    # ✅ Move model to device
    model.to(device)

    print(f"✅ Fine-tuned model loaded successfully from {model_path}")

    # ✅ Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Tokenized sample:", tokenizer("[C][S][Br]", return_tensors="pt").input_ids)

    print(f"✅ Tokenizer loaded successfully from {tokenizer_path}")

    # ✅ Generate text
    input_text = "[C][S][Br]"  # Example SELFIES input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # generated_ids = model.generate(input_ids, max_length=50, num_return_sequences=100, num_beams=200)
    # generated_ids = model.generate(
    #     input_ids,
    #     max_length=100,
    #     num_return_sequences=5,
    #     num_beams=5,  # Reduce beam search to avoid deterministic outputs
    #     temperature=1.2,  # Increase randomness
    #     top_k=50,  # Limit to top 50 tokens
    #     top_p=0.9,  # Use nucleus sampling
    #     do_sample=True  # Enable sampling
    # )


    # generated_ids = model.generate(
    #     input_ids,
    #     max_length=200,
    #     num_return_sequences=5,
    #     do_sample=True,  # ✅ Enable sampling (adds randomness)
    #     temperature=1.5,  # ✅ Increase randomness (higher values = more diverse outputs)
    #     top_k=50,  # ✅ Consider only top 50 most likely next tokens
    #     top_p=0.95,  # ✅ Use nucleus sampling (focus on probable tokens)
    #     repetition_penalty=1.2,  # ✅ Penalize repetitive phrases
    #     num_beams=1  # ✅ Disable beam search (prevents deterministic output)
    # )

    generated_ids = model.generate(
        input_ids,
        max_length=500,
        num_return_sequences=10,
        do_sample=True,  # ✅ Enable sampling (adds randomness)
        temperature=2.5,  # ✅ Increase randomness (higher values = more diverse outputs)
        top_k=50,  # ✅ Consider only top 50 most likely next tokens
        top_p=0.95,  # ✅ Use nucleus sampling (focus on probable tokens)
        repetition_penalty=1.8,  # ✅ Penalize repetitive phrases
        num_beams=1  # ✅ Disable beam search (prevents deterministic output)
    )


    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("Generated Texts:", generated_texts)
