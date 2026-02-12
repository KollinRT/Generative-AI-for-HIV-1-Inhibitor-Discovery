"""
This script is for generating molecules from the pretrained and fine-tuned model for use for evaluating model performance.
"""

from token_mapping import token_mapping
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast


def generate_text(
    model: BartForConditionalGeneration,
    tokenizer: PreTrainedTokenizerFast,
    input_text: str,
    max_length: int = 100,
    num_return_sequences: int = 5,
    num_beams: int = 5,
):
    """
    Args:
        model : BartForConditionalGeneration
            The pretrained BART model used for text generation.

        tokenizer : transformers.PreTrainedTokenizerFast
            Tokenizer corresponding to the BART model.

        input_text : str
            Input prompt or sequence to condition the model generation.

        max_length : int, optional (default=100)
            Maximum length of the generated sequence.

        num_return_sequences : int, optional (default=5)
            Number of distinct sequences to return.

        num_beams : int, optional (default=5)
            Number of beams used in beam search.

    Returns:
        decoded_texts : List[str]
        A list of generated text sequences.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode input text correctly
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
    )

    # Decode generated text correctly
    decoded_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return decoded_texts


# def benchmark_generated_molecules():
# pass


def load_model(model_path: str, device: str = "cuda"):
    """
    Args:
        model_path : str
            Path to the fine-tuned model directory, or a HuggingFace model identifier.

        device : str, optional (default="cuda")
            Device to load the model onto ("cuda", "cpu", or "mps").

    Returns:
        model : BartForConditionalGeneration
            The loaded and device-mapped model.


    """
    model = BartForConditionalGeneration.from_pretrained(
        model_path
    )  # Initialize the model architecture
    # model.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary
    # model = model.to(device)
    return model


def load_tokenizer(tokenizer_path):
    """
    Args:
        tokenizer_path: str
            Path to the tokenizer file directory.
    Returns:
        tokenizer: transformers.PretrainedTokenizerFast

    """

    """Load the tokenizer used for both encoding and decoding."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    import torch
    from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

    # Set paths
    model_path = "./runs/selfies_BART_PRETRAIN_model_4_warmup/model"
    model = load_model(model_path)
    # tokenizer = load_tokenizer("selfies_word_tokenizer_12M")
    tokenizer = load_tokenizer("full_tokenizer_finetune_and_pretrain")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate text
    input_text = "<s>"

    tokens = input_text.split()  # Assumes space-delimited SELFIES tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    input_ids = torch.tensor([ids]).to(device)
    print(input_ids)

    generated_ids = model.generate(
        input_ids,
        max_length=32,
        num_return_sequences=50,
        do_sample=True,  # Enable sampling (adds randomness)
        temperature=1.0,  # Increase randomness (higher values = more diverse outputs)
        top_k=50,  # Consider only top 50 most likely next tokens
        top_p=0.95,  # Use nucleus sampling (focus on probable tokens)
        repetition_penalty=1.0,  # Penalize repetitive phrases
        num_beams=50,  # Disable beam search (prevents deterministic output)
    )

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Apply token mapping to each token before constructing the final SELFIES string
    cleaned_selfies = []
    for text in generated_texts:
        tokens = text.strip().split()  # Token-level split

        mapped_tokens = [
            token_mapping.get(tok, tok) for tok in tokens
        ]  # Use mapping; fallback to original if not found

        selfies_string = "".join(mapped_tokens)
        cleaned_selfies.append(selfies_string)
