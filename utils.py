import os
import torch

def diff_to_string(base_config, other_config):
    diffs = {}
    for key in base_config:
        base_val = base_config.get(key)
        other_val = other_config.get(key)
        if base_val != other_val:
            diffs[key] = other_val
    return diffs

# def encode_differences_to_string(base_model_name, diffs):
def encode_differences_to_string(base_model_name, base_config, other_config):
    diffs = diff_to_string(base_config, other_config)
    parts = [base_model_name]
    for key, value in sorted(diffs.items()):
        safe_key = str(key).upper().replace(" ", "_")
        safe_val = str(value).replace("/", "-").replace(" ", "").replace(":", "-")
        parts.append(f"{safe_key}-{safe_val}")
    return "__".join(parts)


def save_final_model_if_needed(model, save_dir):
    """Ensure the final model is saved if no best version was saved."""
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        print("🟡 No best model saved. Saving final model manually.")
        model.save_pretrained(save_dir)

def write_done_marker(save_dir):
    """Write a flag file to signal training completed."""
    done_path = os.path.join(save_dir, "done.txt")
    with open(done_path, "w") as f:
        f.write("Training complete\n")
    print(f"✅ Written done.txt to {done_path}")

def is_model_done(save_dir):
    """Check if training has already been completed for this model."""
    return os.path.exists(os.path.join(save_dir, "done.txt"))
