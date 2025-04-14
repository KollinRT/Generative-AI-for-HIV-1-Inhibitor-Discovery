def diff_to_string(base_config, other_config):
    diffs = {}
    for key in base_config:
        base_val = base_config.get(key)
        other_val = other_config.get(key)
        if base_val != other_val:
            diffs[key] = other_val
    return diffs

def encode_differences_to_string(base_model_name, diffs):
    parts = [base_model_name]
    for key, value in sorted(diffs.items()):
        safe_key = str(key).upper().replace(" ", "_")
        safe_val = str(value).replace("/", "-").replace(" ", "").replace(":", "-")
        parts.append(f"{safe_key}-{safe_val}")
    return "__".join(parts)
