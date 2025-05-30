"""
This script is for generating molecules from the pretrained and fine-tuned model for use for evaluating model performance.
"""

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
    model_path = "selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw"
    # model = BartForConditionalGeneration.from_pretrained("DataForGen/selfies_BART_pretrained__skip_base__LEARNING_RATE-3e-05__EARLY_STOPPING_PATIENCE-8__EARLY_STOPPING_THRESHOLD-0.0001__LR_SCHED-{'type'-'linear','warmup_ratio'-0.1}__OPTIMIZER-adamw")
    model = load_model(model_path)
    tokenizer = load_tokenizer("selfies_word_tokenizer")

    # ✅ Set device
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ✅ Generate text
    # # input_text = "<s>"  # Example SELFIES input
    # # input_text.
    # # input_text = "[C] [C] [Branch1_1] [O][C][C][N][C][=O][C][C][Ring1][C][C][=O][O]"
    # input_text = "[C] [C] [O]"
    # input_text = "[C] [C] [Branch1_1] [O] [C] [C] [N] [C] [=O] [C] [C] [Ring1] [C] [C] [=O] [O]"
    input_text = "<s>"
    #
    # # 1. Manual split
    # tokens = input_text.split()
    # print("Tokens:", tokens)
    #
    # # 2. Manual conversion
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("IDs:", input_ids)
    #
    # # 3. Decode
    # decoded = tokenizer.decode(input_ids)
    # print("Decoded:", decoded)
    #
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    # print(input_ids)

    tokens = input_text.split()  # Assumes space-delimited SELFIES tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    input_ids = torch.tensor([ids]).to(device)
    print(input_ids)

    generated_ids = model.generate(
        input_ids,
        max_length=500,
        num_return_sequences=100,
        do_sample=True,  # ✅ Enable sampling (adds randomness)
        temperature=1.0,  # ✅ Increase randomness (higher values = more diverse outputs)
        top_k=50,  # ✅ Consider only top 50 most likely next tokens
        top_p=0.95,  # ✅ Use nucleus sampling (focus on probable tokens)
        repetition_penalty=1.0,  # ✅ Penalize repetitive phrases
        num_beams=1  # ✅ Disable beam search (prevents deterministic output)
    )


    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned_selfies = [''.join(t.split()) for t in generated_texts]

    print("Generated Texts:", generated_texts)
    print(cleaned_selfies)