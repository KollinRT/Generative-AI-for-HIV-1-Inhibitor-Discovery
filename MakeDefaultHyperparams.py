import yaml
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    default="FinetuneSpecs.yml",
    help="Path to the configuration YAML file",
)
args = parser.parse_args()

# Default configuration
default_config = OrderedDict(
    [
        ("HIDDEN_SIZE", 768),
        ("TRAIN_BATCH_SIZE", 16),
        ("VALID_BATCH_SIZE", 8),
        ("TRAIN_EPOCHS", 100),
        ("LEARNING_RATE", 0.00005),
        ("WEIGHT_DECAY", 0.01),
        ("MAX_LEN", 128),
        ("ENCODER_LAYERS", 12),
        ("DECODER_LAYERS", 12),
        ("NUM_ENCODER_ATTENTION_HEADS", 12),
        ("NUM_DECODER_ATTENTION_HEADS", 12),
        ("ENCODER_FFN_DIM", 3072),
        ("DECODER_FFN_DIM", 3072),
        ("MAX_POSITION_EMBEDDINGS", 514),
        ("NUM_ATTENTION_HEADS", 12),
        ("NUM_HIDDEN_LAYERS", 8),
        ("TYPE_VOCAB_SIZE", 1),
        ("optimizer", "adam"),
        ("criterion", "crossentropy"),
        ("early_stopping_toggle", True),
        ("early_stopping_threshold", 5.0e-5),
        ("early_stopping_patience", 5),
        ("lr_sched", None),
    ]
)
# TODO: Add early stopping here:
"""
Plan of attack:
1. early_stopping_enable = true/false Boolean
2. early_stopping_threshold = Difference? Float
3. early_stopping_patience = Difference in steps? Int

The first is required to toggle the second or third? Patience is interesting in which we check the number of steps since the last improvement. We just want it to improve more! If it gets worse than the best, we keep stacking up...

    # Early stopping logic
    if early_stopping_enabled:
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter if there's improvement
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break
other metrics:
For Text Generation:
Perplexity:
BLEU (Bilingual Evaluation Understudy):
ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
Distinct-n Metrics:
Human Evaluation:
?
I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across metrics...I think I can check out perplexity? See what other CLMs utilize and look at those. Look at how they are utilized in training and that'll help me...
Maybe refactor into a class-based early stopping but only if it is reusable across keys...
"""


# Path to the additional configurations YAML file
additional_configs_file = args.file  # 'FinetuneSpecs.yml'

# Load the additional configurations
with open(additional_configs_file, "r") as file:
    additional_configs = yaml.safe_load(file)

# Extract the molecular properties to filter
molecular_properties_to_filter = additional_configs.get(
    "molecular_properties_to_filter", {}
)

# Extract only the model names
model_names = list(molecular_properties_to_filter.keys())

# Create a new configuration that includes the default configuration for each model name
combined_config = {model_name: default_config.copy() for model_name in model_names}

# This adds the BART: as the primary key...
# Nest the combined configuration under the 'BART' key
final_config = {"BART": combined_config}


# Define a custom representer for OrderedDict to avoid !!python/object/apply:collections.OrderedDict
def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, represent_ordereddict)

# Save the combined configuration to a new YAML file
output_filename = "combined_config.yml"
with open(output_filename, "w") as output_file:
    yaml.dump(final_config, output_file, default_flow_style=False, sort_keys=False)
print(f"Combined configuration saved to {output_filename}")


### Non-ordered dict way
# # Default configuration
# default_config = {
#     "HIDDEN_SIZE": 768,
#     "TRAIN_BATCH_SIZE": 16,
#     "VALID_BATCH_SIZE": 8,
#     "TRAIN_EPOCHS": 100,
#     "LEARNING_RATE": 0.00005,
#     "WEIGHT_DECAY": 0.01,
#     "MAX_LEN": 128,
#     "ENCODER_LAYERS": 12,
#     "DECODER_LAYERS": 12,
#     "NUM_ENCODER_ATTENTION_HEADS": 12,
#     "NUM_DECODER_ATTENTION_HEADS": 12,
#     "ENCODER_FFN_DIM": 3072,
#     "DECODER_FFN_DIM": 3072,
#     "VOCAB_SIZE": 30000,
#     "MAX_POSITION_EMBEDDINGS": 514,
#     "NUM_ATTENTION_HEADS": 12,
#     "NUM_HIDDEN_LAYERS": 8,
#     "TYPE_VOCAB_SIZE": 1,
# }

# # Path to the additional configurations YAML file
# additional_configs_file = 'FinetuneSpecs.yml'

# # Load the additional configurations
# with open(additional_configs_file, 'r') as file:
#     additional_configs = yaml.safe_load(file)

# # Extract the molecular properties to filter
# molecular_properties_to_filter = additional_configs.get('molecular_properties_to_filter', {})

# # Extract only the model names
# model_names = list(molecular_properties_to_filter.keys())

# # Create a new configuration that includes the default configuration for each model name
# combined_config = {
#     model_name: default_config.copy() for model_name in model_names
# }

# # Save the combined configuration to a new YAML file
# output_filename = "combined_config.yml"
# with open(output_filename, 'w') as output_file:
#     yaml.dump(combined_config, output_file, default_flow_style=False, sort_keys=False)
# print(f"Combined configuration saved to {output_filename}")
