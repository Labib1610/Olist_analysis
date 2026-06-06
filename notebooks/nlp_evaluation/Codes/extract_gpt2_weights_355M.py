"""
Run this ONCE from the terminal — NOT inside Jupyter.

    python extract_gpt2_weights_355M.py

It downloads the GPT-2 355M checkpoint via TensorFlow, flattens every
tensor into a single .npz file, then exits.  After that you never need
TensorFlow again; the notebook loads only PyTorch.

Usage:
    cd <your project folder>          # same dir as the notebook
    python extract_gpt2_weights_355M.py
"""

import os, json, sys
import numpy as np
import requests
from tqdm import tqdm

# ── 1. Download raw checkpoint files ───────────────────────────────────────
MODEL_SIZE  = "355M"
MODELS_DIR  = "gpt2"
MODEL_DIR   = os.path.join(MODELS_DIR, MODEL_SIZE)
BASE_URL    = "https://openaipublic.blob.core.windows.net/gpt-2/models"
FILENAMES   = [
    "checkpoint", "encoder.json", "hparams.json",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index",
    "model.ckpt.meta", "vocab.bpe",
]
OUTPUT_NPZ  = os.path.join(MODELS_DIR, "gpt2_355M_weights.npz")

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, dest):
    r = requests.get(url, stream=True, verify=False)
    size = int(r.headers.get("content-length", 0))
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  already up-to-date: {dest}")
        return
    with tqdm(total=size, unit="iB", unit_scale=True, desc=os.path.basename(dest)) as bar:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024):
                bar.update(len(chunk))
                f.write(chunk)

print("=== Downloading GPT-2 355M checkpoint ===")
for fn in FILENAMES:
    download_file(f"{BASE_URL}/{MODEL_SIZE}/{fn}", os.path.join(MODEL_DIR, fn))

# ── 2. Load via TensorFlow (isolated to this script only) ──────────────────
print("\n=== Loading TF checkpoint & flattening weights ===")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # silence TF noise
os.environ["CUDA_VISIBLE_DEVICES"]  = ""           # force TF to use CPU only
                                                    # so it can't touch the GPU
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

settings    = json.load(open(os.path.join(MODEL_DIR, "hparams.json")))
ckpt_path   = tf.train.latest_checkpoint(MODEL_DIR)

flat = {}   # key -> numpy array, e.g. "h0|attn|c_attn|w"
for name, _ in tf.train.list_variables(ckpt_path):
    arr  = np.squeeze(tf.train.load_variable(ckpt_path, name))
    # strip leading "model/" and replace "/" with "|" (npz keys can't have /)
    key = name.split("/", 1)[1].replace("/", "|")
    flat[key] = arr
    print(f"  {name:50s}  {arr.shape}")

# also save hparams so the notebook can reconstruct BASE_CONFIG
flat["__n_layer__"] = np.array(settings["n_layer"])
flat["__n_head__"]  = np.array(settings["n_head"])
flat["__n_embd__"]  = np.array(settings["n_embd"])

np.savez(OUTPUT_NPZ, **flat)
print(f"\n✓  Saved {len(flat)} tensors → {OUTPUT_NPZ}")
print(f"   n_layer={settings['n_layer']}, n_head={settings['n_head']}, n_embd={settings['n_embd']}")
print("You can now delete TensorFlow from this venv if you like.")
