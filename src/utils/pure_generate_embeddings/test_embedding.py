#!/usr/bin/env python3
import os
import time
from sentence_transformers import SentenceTransformer
dir_now = os.path.dirname(os.path.abspath(__file__))
dir_text = dir_now + "/../../../" + "data/raw/embeddings/"
# ─── Configuration ─────────────────────────────────────────────────────────────

# List the embedding models to test here
MODEL_NAMES = [
    # "BAAI/bge-small-en-v1.5", #slow 0.6
    # "sentence-transformers/all-MiniLM-L6-v2", #slow 0.54
    "sentence-transformers/all-mpnet-base-v2", #fast 0.45
    # "all-distilroberta-v1", # slow 0.55
    # "paraphrase-MiniLM-L3-v2", #fast 0.45
]

# Path to the single text file to test

TEST_FILE_PATH = os.path.expanduser(dir_text+"/test/20447626similar/retrieved_29925814.txt")

# ─── Helper ────────────────────────────────────────────────────────────────────

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    text = load_text(TEST_FILE_PATH)
    print(f"Testing file: {TEST_FILE_PATH!r}\n")

    for model_name in MODEL_NAMES:
        print(f"→ Loading model `{model_name}`…", end="", flush=True)
        start_load = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_load
        print(f" done in {load_time:.2f}s")

        # embed
        start_enc = time.time()
        emb = model.encode(text, convert_to_numpy=True)
        enc_time = time.time() - start_enc

        print(f"  • Embedding size: {emb.shape}")
        print(f"  • Encoding time:  {enc_time:.2f}s\n")

if __name__ == "__main__":
    main()
