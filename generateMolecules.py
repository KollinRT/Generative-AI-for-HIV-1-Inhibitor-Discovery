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

def generate_text(model, tokenizer, input_text, max_length=50, num_return_sequences=1, num_beams=4):
    """Generate molecules using the model and tokenizer."""
    input_ids = tokenizer.encode(input_text, return_tensors='pt').ids
    input_ids = torch.tensor([input_ids]).to(model.device)
    
    outputs = model.generate(input_ids,
                             max_length=max_length,
                             num_return_sequences=num_return_sequences,
                             num_beams=num_beams,
                             early_stopping=True,
                             no_repeat_ngram_size=2)

    # Decode and return each generated sequence
    generated_texts = [
        tokenizer.decode(output.tolist(), skip_special_tokens=True)
        for output in outputs
    ]
    return generated_texts

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

if __name__ == "__main__":
    # Specify the paths to the model and tokenizer
    model_path = f'./selfies_BART_finetuned_{key}.pth'  # Path to the saved state dictionary
    tokenizer_path = 'facebook/bart-base'  # Use the pre-trained tokenizer from Hugging Face
    # TODO: New, update the tokenizer_path above!
    """
    #     tokenizer_path = './data/bpe/bpe.json'
    Something like that above.... data _{key} yeah...
    """

    # Set the device to use (either 'cuda' or 'cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer
    model = load_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    # Input text for generation
    input_text = "[C][S][Br]" # Is this utilized? I do not see it!

    # Generate text
    generated_texts = generate_text(model,
                                    tokenizer,
                                    input_text,
                                    max_length=1000,
                                    num_return_sequences=2000,
                                    num_beams=2000)

    # Print generated texts
    for text in generated_texts:
        print("Generated Text:", text)