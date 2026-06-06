# ----------------------------------------
# Imports
# ----------------------------------------

import os
os.environ["HF_HOME"] = "/scratch/cs26d002/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import random
import json
import torch
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------
# Redirect HuggingFace cache to scratch
# MUST be set before any HF imports/calls
# ----------------------------------------

USERNAME = "cs26d002"  # your DGX username

# os.environ["HF_HOME"] = f"/scratch/{USERNAME}/hf_cache"

# ----------------------------------------
# Paths
# ----------------------------------------

PROJECT_DIR = f"/scratch/{USERNAME}/NLA_Phi3_khanacademy"

LAST_ACTIVATIONS_DIR = os.path.join(
    PROJECT_DIR,
    "LastActivation"
)

MEAN_ACTIVATIONS_DIR = os.path.join(
    PROJECT_DIR,
    "MeanActivation"
)

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(LAST_ACTIVATIONS_DIR, exist_ok=True)
os.makedirs(MEAN_ACTIVATIONS_DIR, exist_ok=True)

DATASET_JSON_PATH = os.path.join(
    PROJECT_DIR,
    "dataset.json"
)

# ----------------------------------------
# Reproducibility
# ----------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------------------
# Enable TF32
# ----------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("TF32 acceleration enabled.")

# ----------------------------------------
# Load WikiText
# ----------------------------------------

print("Loading KhanAcademy dataset...")

dataset = load_dataset(
    "HuggingFaceTB/cosmopedia",
    "khanacademy",
    split="train"
)

texts = []

for item in dataset:
    text = item["text"].strip()
    if len(text.split()) > 20:
        texts.append(text)

random.shuffle(texts)

texts = texts[:5000]

print(f"Total Samples: {len(texts)}")

# ----------------------------------------
# Load Phi-3
# ----------------------------------------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Phi-3 loaded successfully.")
print(f"Model device: {next(model.parameters()).device}")

# ----------------------------------------
# Config
# ----------------------------------------

LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

# ----------------------------------------
# Prompt Builder
# ----------------------------------------

def build_prompt(text):

    prompt = f"""
You are a semantic labeling engine.

Read the given text and output ONLY a concise semantic topic label.

Rules:
- Maximum 10 words
- No punctuation
- No explanations
- No full sentences
- Output only the semantic label
- Capture the main entities events or topics
- Avoid generic labels

Examples:
Ancient Roman political history
Neural network optimization methods
Egyptian mythology divine kingship conflict
Video game franchise development
Michael Jordan basketball dominance

TEXT:
{text}

LABEL:
"""
    return prompt.strip()

# ----------------------------------------
# Containers
# ----------------------------------------

dataset_entries = []

activation_list_last = {
    layer: []
    for layer in LAYERS
}

activation_list_mean = {
    layer: []
    for layer in LAYERS
}

# ----------------------------------------
# Main Processing Loop
# ----------------------------------------

print("Starting processing loop...")

for idx, text in enumerate(tqdm(texts)):

    try:

        prompt = build_prompt(text)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        # ----------------------------
        # Forward pass for activations
        # ----------------------------

        with torch.no_grad():

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).float()

        for layer in LAYERS:

            target_hidden = hidden_states[layer].float()

            # Last token activation
            last_activation = (
                target_hidden[:, -1, :]
                .squeeze(0)
                .cpu()
            )

            # Attention-weighted mean pooling
            mean_activation = (
                (target_hidden * mask).sum(dim=1)
                / mask.sum(dim=1)
            ).squeeze(0).cpu()

            activation_list_last[layer].append(
                last_activation
            )

            activation_list_mean[layer].append(
                mean_activation
            )

        # ----------------------------
        # Generate label
        # ----------------------------

        with torch.no_grad():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        generated_only_ids = generated_ids[0][input_len:]

        description = tokenizer.decode(
            generated_only_ids,
            skip_special_tokens = True
        ).strip()

        description = description.split("\n")[0].strip()

        #generated_text = tokenizer.decode(
        #    generated_ids[0],
        #    skip_special_tokens=True
        #)

        # Extract only the generated label
        # description = generated_text[len(prompt):]
        # description = description.strip().split("\n")[0].strip()

        dataset_entries.append({
            "text": text,
            "phi3_description": description
        })

        # Print progress every 100 samples
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/1000 samples")
            print(f"  Last label: {description}")

    except Exception as e:
        print(f"Error at sample {idx}: {e}")
        # Still append a placeholder so indices stay aligned
        dataset_entries.append({
            "text": text,
            "phi3_description": "generation_failed"
        })
        for layer in LAYERS:
            # Append zero vectors to keep alignment
            activation_list_last[layer].append(
                torch.zeros(3072)
            )
            activation_list_mean[layer].append(
                torch.zeros(3072)
            )

# ----------------------------------------
# Save dataset.json
# ----------------------------------------

with open(DATASET_JSON_PATH, "w") as f:
    json.dump(dataset_entries, f, indent=2)

print(f"dataset.json saved → {DATASET_JSON_PATH}")
print(f"Total entries: {len(dataset_entries)}")

# ----------------------------------------
# Save activation tensors
# ----------------------------------------

print("Saving activation tensors...")

for layer in LAYERS:

    last_tensor = torch.stack(
        activation_list_last[layer]
    )

    mean_tensor = torch.stack(
        activation_list_mean[layer]
    )

    last_path = os.path.join(
        LAST_ACTIVATIONS_DIR,
        f"layer_{layer}.pt"
    )

    mean_path = os.path.join(
        MEAN_ACTIVATIONS_DIR,
        f"layer_{layer}.pt"
    )

    torch.save(last_tensor, last_path)
    torch.save(mean_tensor, mean_path)

    print(f"Layer {layer} saved.")
    print(f"  Last shape: {last_tensor.shape}")
    print(f"  Mean shape: {mean_tensor.shape}")

# ----------------------------------------
# Verification
# ----------------------------------------

print("\n===== VERIFICATION =====")
print(f"Total dataset entries: {len(dataset_entries)}")
print(f"Sample entry: {dataset_entries[0]}")

for layer in LAYERS:
    print(
        f"Layer {layer} → "
        f"Last: {len(activation_list_last[layer])} | "
        f"Mean: {len(activation_list_mean[layer])}"
    )

print("All done.")

