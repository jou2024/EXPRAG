#!/bin/bash

models=(
      # Medical 
      # "TsinghuaC3I/Llama-3-70B-UltraMedical"
      #  "baichuan-inc/Baichuan-M1-14B-Instruct"

       "4o-mini"
      # "3.5"
      # "4o"

      # Mistral
      # 'mistralai/Mistral-7B-Instruct-v0.3'

      # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


)

# Define paths to the Python scripts
MAIN_SCRIPT_PATH="main.py"
MED_DIAG_SCRIPT_PATH="main_medications_diagnosis.py"
DATA_BASE_PATH="/../../DataSets/mimic-iv/mimic4-note.db"

for model in "${models[@]}"; do

    # Print the parameters being used
    echo "Running with the following parameters:"
    echo "LLM: $model"

    # Running main.py with the specified model; max 400 for full size (non-demo)
    echo "Running $MAIN_SCRIPT_PATH with $model..."
    python3 $MAIN_SCRIPT_PATH \
          --llm "$model" \
          --dataset "benchmark_choice" \
          --notes "auto_merging_combine_10chunk_" \
          --num_questions 20 \
          --is_rag True \
          --qa_file "/../data/generated_qa_json/DischargeQA_v2_instruction-DEMO.json" \
          --prompt_file "/../data/processed/our_benchmark_prompt/prompt_instruction_rag_combine.txt" \
          --database_file "$DATA_BASE_PATH" \
          --similar_patients_map_file "/../data/find_similar_patients/similar_patients_scores_0-2000_combined.csv" \
          --tmp_prompt_for_similar_patients_file "/../data/processed/our_benchmark_prompt/question_frame_for_similar_patients.txt" \
          --tmp_prompt_rag_query_with_similar_patients_background_file "/../data/processed/our_benchmark_prompt/question_background_similar_patients.txt"

    # Running main_medications_diagnosis.py with the specified model on Medication; max 444 for full size (non-demo)
    echo "Running $MED_DIAG_SCRIPT_PATH with $model on Medication..."
    python3 $MED_DIAG_SCRIPT_PATH \
          --llm "$model" \
          --dataset "benchmark_medications" \
          --notes "auto_merging_combine_10chunk_" \
          --num_questions 20 \
          --is_rag True \
          --qa_file "/../data/generated_medications_qa_json/DischargeQA_v2_medications-DEMO.json" \
          --prompt_file "/../data/processed/our_benchmark_prompt/prompt_medication_rag_combine.txt" \
          --database_file "$DATA_BASE_PATH" \
          --similar_patients_map_file "/../data/find_similar_patients/similar_patients_scores_0-2000_combined.csv" \
          --tmp_prompt_for_similar_patients_file "/../data/processed/our_benchmark_prompt/question_frame_for_similar_patients.txt" \
          --tmp_prompt_rag_query_with_similar_patients_background_file "/../data/processed/our_benchmark_prompt/question_background_similar_patients.txt"


    # Running main_medications_diagnosis.py with the specified model on Diagnosis; max 436 for full size (non-demo)
    echo "Running $MED_DIAG_SCRIPT_PATH with $model on Diagnosis..."
    python3 $MED_DIAG_SCRIPT_PATH \
          --llm "$model" \
          --dataset "benchmark_diagnosis" \
          --notes "auto_merging_combine_10chunk_" \
          --num_questions 20 \
          --is_rag True \
          --qa_file "/../data/generated_diagnosis_qa_json/DischargeQA_v2_diagnosis-DEMO.json" \
          --prompt_file "/../data/processed/our_benchmark_prompt/prompt_diagnosis_rag_combine_2.txt" \
          --database_file "$DATA_BASE_PATH" \
          --similar_patients_map_file "/../data/find_similar_patients/similar_patients_scores_0-2000_combined.csv" \
          --tmp_prompt_for_similar_patients_file "/../data/processed/our_benchmark_prompt/question_frame_for_similar_patients.txt" \
          --tmp_prompt_rag_query_with_similar_patients_background_file "/../data/processed/our_benchmark_prompt/question_background_similar_patients.txt"

done

