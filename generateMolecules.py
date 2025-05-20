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

# def load_model(model_path, device='cuda'):
#     """Load the pre-trained BART model."""
#     model = BartForConditionalGeneration.from_pretrained(model_path)  # Initialize the model architecture
#     model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary
#     model = model.to(device)
#     model.eval()  # Set model to evaluation mode
#     return model




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
from tokenizers import Tokenizer
import os

# def load_tokenizer(tokenizer_path):
#     # ✅ Check if tokenizer_path is a directory
#     if os.path.isdir(tokenizer_path):
#         # Look for a tokenizer JSON file inside
#         possible_files = ["bpe.json", "tokenizer.json", "vocab.json"]
#         for file in possible_files:
#             full_path = os.path.join(tokenizer_path, file)
#             if os.path.exists(full_path):
#                 tokenizer_path = full_path
#                 break
#         else:
#             raise FileNotFoundError(f"❌ No valid tokenizer file found in {tokenizer_path}")
#
#     # ✅ Load the tokenizer from the correct file
#     return Tokenizer.from_file(tokenizer_path)

def load_model(model_path, device='cuda'):
    """Load the pre-trained BART model."""
    model = BartForConditionalGeneration.from_pretrained(model_path)  # Initialize the model architecture
    # model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary
    # model = model.to(device)
    return model

def load_tokenizer(tokenizer_path):
    """Load the tokenizer used for both encoding and decoding."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    import torch
    from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

    # ✅ Set paths
    model_path = "DataForGen/selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw"
    # model = BartForConditionalGeneration.from_pretrained("DataForGen/selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw")
    model = load_model(model_path)
    tokenizer = load_tokenizer("DataForGen/selfies_word_tokenizer")

    # ✅ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ✅ Generate text
    input_text = "[C][S][Br]"  # Example SELFIES input
    # input_text.
    # input_text = "[C][C][Branch1_1][O][C][C][N][C][=O][C][C][Ring1][C][C][=O][O]"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(
        input_ids,
        max_length=500,
        num_return_sequences=100,
        do_sample=True,  # ✅ Enable sampling (adds randomness)
        temperature=0.5,  # ✅ Increase randomness (higher values = more diverse outputs)
        top_k=50,  # ✅ Consider only top 50 most likely next tokens
        top_p=0.95,  # ✅ Use nucleus sampling (focus on probable tokens)
        repetition_penalty=1.8,  # ✅ Penalize repetitive phrases
        num_beams=1  # ✅ Disable beam search (prevents deterministic output)
    )


    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned_selfies = [''.join(t.split()) for t in generated_texts]

    print("Generated Texts:", generated_texts)
    print(cleaned_selfies)