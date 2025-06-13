import os
import pickle
import numpy as np
from tqdm import tqdm
from .index import Indexer
from datetime import datetime

dir_now = os.path.dirname(os.path.abspath(__file__))

# Path to the folder containing text files
raw_data_relative_path = "data/raw/embeddings/all_batch/"
text_folder_path = dir_now + "/../../../" + raw_data_relative_path


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIR = dir_now + "/../../../" + "data/processed/embeddings/embed_pure_all_batch/"
# EMBED_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# EMBEDDING_DIR = dir_now + "/../../../" + "data/processed/embeddings/embed_pure_all_batch_paraphrase-MiniLM-L3-v2"
TOP_K = 10
BATCH_SIZE = 2048

example_query = '''
        "background": "a 44 years old male with a history of recent STEMI and stent placement in the LAD. He experienced bilateral arm tightness and a feeling of uneasiness, but no chest pain or dyspnea. He was pain-free upon admission and wanted to go home. His past medical history includes CAD, dyslipidemia, and hypertension. Family history includes a father who died of PE from DVT and a brother with blood clot issues.",
        "discharge_diagnosis_options": {
            "A": "Pain in limb",
            "B": "Acute myocardial infarction",
            "C": "Hyperlipidemia",
            "D": "Hypertension",
            "E": "Percutaneous transluminal coronary angioplasty status",
            "F": "Family history of ischemic heart disease",
            "G": "CAD with previous stent placement",
            "H": "Personal history of tobacco use"
        },
        "question": "Which diagnoses should be documented into the patient's discharge diagnosis?",
'''

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Function to ensure directories exist
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Save embeddings with checkpoints
def save_embeddings(file_path, embedding, output_dir):
    relative_path = os.path.relpath(file_path, start=raw_data_relative_path)
    # print("output_dir", output_dir)
    # print("relative_path", relative_path)
    embedding_path = os.path.join(output_dir, relative_path + ".pkl")
    ensure_dir(os.path.dirname(embedding_path))
    with open(embedding_path, "wb") as f:
        pickle.dump(embedding, f)

# Generate and store embeddings
def generate_embeddings(input_dir, output_dir):
    # Initialize model

    for folder in tqdm(os.listdir(input_dir), desc="Processing folders"):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        start_time = datetime.now()
        print(f"Start processing folder: {folder} at {start_time}")

        for file in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder}", leave=False):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                continue

            embedding_path = os.path.join(output_dir, folder, file + ".pkl")
            if os.path.exists(embedding_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            embedding = embed_model.encode(text, convert_to_numpy=True)
            save_embeddings(file_path, embedding, output_dir)

        end_time = datetime.now()
        print(f"Finished processing folder: {folder} at {end_time} to {output_dir}")
        print(f"Time taken for folder {folder}: {end_time - start_time}")

# Find top-k relevant files based on query
def search_query(query, embedding_dir, top_k=TOP_K):
    query_embedding = embed_model.encode(query, convert_to_numpy=True)
    indexer = Indexer(vector_sz=query_embedding.shape[0])

    for folder in tqdm(os.listdir(embedding_dir), desc="Scan embedding folders"):
        folder_path = os.path.join(embedding_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        embeddings = []
        ids = []

        for file in os.listdir(folder_path):
            if file == ".ipynb_checkpoints":
                continue
            file_path = os.path.join(folder_path, file)
            with open(file_path, "rb") as f:
                embedding = pickle.load(f)
                embeddings.append(embedding)
                ids.append(file)

        embeddings = np.array(embeddings)
        indexer.index_data(ids, embeddings)
        # current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        # print(f"{current_time} {folder} is scanned")
    
    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{current_time} Try to find result")
    results = indexer.search_knn(np.array([query_embedding]), top_k)
    print(f"{current_time} Get result size {len(results[0][0])}")
    return results

# Main execution
if __name__ == "__main__":
    INPUT_DIR = text_folder_path
    ensure_dir(EMBEDDING_DIR)

    # Step 1: Generate embeddings
    generate_embeddings(INPUT_DIR, EMBEDDING_DIR)

    # Step 2: Query embedding and search
    # query = example_query
    # results = search_query(query, EMBEDDING_DIR)

    # print("Top-k results:")
    # for files, scores in results:
    #     for file, score in zip(files, scores):
    #         print(f"File: {file}, Score: {score}")
