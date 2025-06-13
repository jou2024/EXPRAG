# EXPRAG 
This is the repository for publication "Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA"
 
Datasets (3 QA tasks, 1280 QA pairs), embedding files for 320,000 patients summaries file will be uploaded to https://physionet.org/ as DischargeQA

## Logs
Updated DEMO data, prompts, utils and experiments main functions 
Updated codes for embeddings generation. 
Updated codes, filtered id data and similarity map, everything for generating our benchmark.

## Setup

### Library
**Ensure Python dependencies are installed.**  
   You need the packages in requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
Note: the version of llama-index-core related may have conflict which may need adjustment.

### Dataset MIMIC-IV csv and db (SQL)

MIMIC-IV:
All csv files (eg. admissions.csv) of mimic-iv: in data/raw/mimic-iv-link/all_csv/
Database files should be placed here: 
data/raw/mimic-iv-link/mimic4-note.db

MIMIC-IV-Note:
data/raw/mimic-iv-link/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv
data/raw/mimic-iv-link/mimic4-note.db

Note: some test scripts need path in .env file below:
DB_PATH=<Your MIMIC4 note DB Path>
CSV_DB_PATH=<Your MIMIC4 note discharge.csv Path>

### Keys
**Set up your `.env` file for Keys** in `src/utils/`:
```plaintext
OPENAI_API_KEY="<Your own key>"
HF_TOKEN="<Your Huggingface token>"
DB_PATH="<Your mimic4-note.db path>"
CSV_DB_PATH="<Your mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv path>"
```

## Embeddings
If you have prepared your DBs/CSVs: 
Before your baseline please follow: Extract - Batch - Generate (test) in `src/utils/pure_generate_embeddings`

## Run experiments
**Note** only DEMO datasets (20 QA pairs for each task) are public
**RAG: Doc/Text-Based EHR-Based**
`cd src`
`./run_vllm_rag.sh`

**No-RAG: Direct-Ask**
`cd src`
`./run_vllm_direct_ask.sh`

# Our benchmark: DischargeQA

Only DEMO are ready in `data/generated*/`
Full datasets are still under review