from utils.llm_utils import LLMAgent
from utils.prompt_utils import QAPromptLoader
from utils.sql_utils import SQLiteDB, save_sql_result
from utils.rag_utils import RAGTool
from utils.txt_file_combiner import TxtFileCombiner
from utils.pure_generate_embeddings.generate_embeddings import search_query

from  utils.gpu_utils import print_gpu_utilization
import shutil
import os
import dotenv
import csv
from typing import List, Tuple
import pandas as pd
from datetime import datetime


# Disable or enable background
BENCHMARK_CHOICE_PLACEHOLDER = {
    'background':'background',
    'A':'A',
    'B':'B',
    'C':'C',
    'D':'D',
    'E':'E'
}

BENCHMARK_MEDICATIONS_PLACEHOLDER ={
    'background': 'background',
    'discharge_medications_options':'discharge_medications_options'
}

BENCHMARK_DIAGNOSIS_PLACEHOLDER = {
    'background': 'background',
    'discharge_diagnosis_options': 'discharge_diagnosis_options'
}


script_dir = os.path.dirname(os.path.abspath(__file__))
discharge_summary_dir = script_dir+"/../tmp/discharge_summary/"
env_path = script_dir + "/utils/.env"
_ = dotenv.load_dotenv(env_path)
dir_now = script_dir

EMBEDDING_SETTING = "" # default as empty for BAAI
# EMBEDDING_SETTING = "_paraphrase-MiniLM-L3-v2" # set embedding here embed_pure_all_batch
# EMBEDDING_SETTING = "_paraphrase-MiniLM-L3-v2"

ROOT_DIR = os.path.join(dir_now, "../")
EMBEDDING_DIR = os.path.normpath(os.path.join(ROOT_DIR, f"data/processed/embeddings/embed_pure_all_batch{EMBEDDING_SETTING}/"))

# 0.  SQL Template
template_sql = '''SELECT text
        FROM discharge
        where hadm_id = {patient_id}
        LIMIT 10;'''

CSV_FIELD_ID = "hadm_id"
CSV_FIELD_TEXT = "text"

CSV_DB_PATH = os.getenv("CSV_DB_PATH")


class EHRAGent:
    def __init__(self, llm="3.5"):
        self.llm_agent = None
        self.llm_version = llm
        self.prompt_loader = None
        self.sql_operator = None

        self.dataset = None

        self.extra_prompt_loader = None
        self.tmp_prompt_for_similar_patients = None

        self.database = None
        self.rag_tool = None
        self.max_iteration = 0
        self._is_qa_loaded = False

        self.csv_database = None
        self.csv_database_loaded = False

        self.re_ranker_template_file = None
        self.embedding_scan_ranker = False

        self.placeholder = None

        self.similar_map_topk = 30
        # RAG framework setting
        self.rag_node_top_n = 10 # 10 for OpenAi models as no concern for token limit. Default as 6
        self.set_rag_info_from_similar_patients = "combine"  # separate or combine. Not used in ablation study

        # Ablation study
        self.set_priority_of_similar_patients = "equal" # "prioritized" or "equal" or "low_priority"

        self._is_scored_similar_maps = True  # Default

    def set_dataset(self, dataset_name):
        self.dataset = dataset_name
        if dataset_name == "benchmark_medications":
            self.placeholder = BENCHMARK_MEDICATIONS_PLACEHOLDER
        elif dataset_name == "benchmark_diagnosis":
            self.placeholder = BENCHMARK_DIAGNOSIS_PLACEHOLDER
        else:
            # default benchmark: instruction
            self.placeholder = BENCHMARK_CHOICE_PLACEHOLDER

    def set_max_iteration(self, max_iteration):
        self.max_iteration = max_iteration

    def load_database(self, db_path):
        if self.sql_operator is None:
            self.sql_operator = SQLiteDB
            self.database = self.sql_operator(db_path)

    def load_questions(self, qa_file, template_file, knowledge_file):
        if not self.prompt_loader:
            self.prompt_loader = QAPromptLoader(qa_file, template_file, knowledge_file)
            self._is_qa_loaded = True
            if self.max_iteration == 0:
                self.max_iteration = len(self.prompt_loader.questions_answers)
                print("No max iteration setting. Iterate all " + str(self.max_iteration) + "questions")
    
    def load_csv_database(self, csv_db_path=CSV_DB_PATH):
        print("Loading csv database")
        self.csv_database_loaded = True
        self.csv_database = pd.read_csv(csv_db_path, dtype=str)
    
    def get_retrieved_file(
        self,
        patient_id: str | int,
        target_folder,
        easy_read=True,
    ) -> str:
        """
        Returns the path to 'retrieved_{patient_id}.txt' under
        similar_patients/{target_patient_id}similar/.
        Creates or retrieves from CSV if missing.
        """
        if not self.csv_database_loaded:
            self.load_csv_database()

        pid = str(patient_id)

        # Build absolute target directory
        target_dir = os.path.abspath(
            os.path.join(script_dir, target_folder)
        )
        os.makedirs(target_dir, exist_ok=True)

        filename = f"retrieved_{pid}.txt"
        target_path = os.path.join(target_dir, filename)

        # 1) If already on disk, return it
        if os.path.exists(target_path):
            print(f" Found {filename}")
            return target_path

        # 2) Otherwise, find in DataFrame
        row = self.csv_database[self.csv_database[CSV_FIELD_ID] == pid]
        if row.empty:
            raise FileNotFoundError(
                f"No entry for hadm_id={pid} in CSV"
            )

        text = row.iloc[0][CSV_FIELD_TEXT]
        if easy_read:
            text = text.replace("\\n", "\n")

        # 3) Write it out
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {target_path}")
        return target_path
    
    def output_prompt(self, qa_file, template_file, knowledge_file, update_index=True):
        # Note: this knowledge_file is for global usage; once the loader is ready, it will not be used
        if not self._is_qa_loaded:
            self.load_questions(qa_file, template_file, knowledge_file)
        if self.dataset == "benchmark_choice" or self.dataset == "benchmark_medications" or self.dataset == "benchmark_diagnosis":
            prompt = self.prompt_loader.next_prompt(
                self.placeholder,
                update_index
            )
        else:
            prompt = self.prompt_loader.next_prompt({'patient_id': 'patient_id'}, update_index)
        return prompt

    def output_response(self, prompt):
        if not self.llm_agent:
            self.llm_agent = LLMAgent(llm=self.llm_version)
        response = self.llm_agent.send_msg(prompt)
        return response


    def output_rag_result(self, query_text, file_path, method="sentence", index_relative_path="../tmp/sentence_index"):
        if self.set_rag_info_from_similar_patients == "combine":
            # Only one time RAG: use the designed one
            self.rag_tool = RAGTool(llm=self.llm_version, top_k=self.rag_node_top_n)
        else:
            # Use 3.5 as the default rag for separate patient
            self.rag_tool = RAGTool(top_k=self.rag_node_top_n)
        self.rag_tool.load_documents([file_path])
        print_gpu_utilization("Loaded files")
        # open and read the file:
        with open(file_path) as f:
            if (f.read()) == "[[]]":
                print(f" Skip! Empty [[]] file at {file_path}")
                return "", None

        if method == "sentence":
            structure_response, content = self.rag_tool.rag_sentence_window_retrival(query_text, index_relative_path)
        elif method == "auto_merging":
            structure_response, content = self.rag_tool.rag_auto_merging(query_text, index_relative_path)
        else:
            structure_response, content = self.rag_tool.run_rag(retrieve_method=method,
                                                               query_text=query_text)
        print_gpu_utilization("Got response_text")
        response_text  = structure_response if method == "contriever" or method == "specter" else structure_response.response
        self.rag_tool = None
        print_gpu_utilization("Cleaned self rag_tool")
        return response_text, content

    def rerank_similar_patients(self, all_criteria, input_patient_row, input_top_k):
        if self.dataset == "benchmark_choice":
            all_criteria = [f"top{self.similar_map_topk}_procedures", f"top{self.similar_map_topk}_prescriptions", f"top{self.similar_map_topk}_diagnoses"]
        elif self.dataset == "benchmark_medications":
            all_criteria = [f"top{self.similar_map_topk}_prescriptions", f"top{self.similar_map_topk}_diagnoses", f"top{self.similar_map_topk}_procedures"]
        elif self.dataset == "benchmark_diagnosis":
            all_criteria = [f"top{self.similar_map_topk}_diagnoses", f"top{self.similar_map_topk}_prescriptions", f"top{self.similar_map_topk}_procedures"]

        def _extract_ids_and_scores(patient_row, criterion):
            """
            Extracts IDs and optionally scores from the patient row based on the given criterion.
            # Returns a list of tuples (id, score) or (id, None) if no scores are present.
            """
            similar_patients = []
            if criterion in patient_row:
                similar_patients_str = patient_row[criterion].values[0]
                if pd.notna(similar_patients_str):
                    for entry in similar_patients_str.split(','):
                        if self._is_scored_similar_maps:
                            entry = entry.split(':')[0]
                        similar_patients.append(entry)
            return similar_patients

        def _unique_top_k(similar_patients, top_k):
            """
            Extracts the top_k unique patient IDs, preserving the order and including scores if present.
            """
            unique_patients = list(dict.fromkeys(similar_patients))[:top_k]
            return unique_patients

        def _equal_priority(criteria, patient_row, top_k):
            """
            Combines patients from each criterion equally, aiming to select top_k/3 from each criterion.
            """
            equal_similar_patients = []
            num_per_criterion = max(1, top_k // len(criteria))

            # Collect patients from each criterion
            for criterion in criteria:
                similar_patients = _extract_ids_and_scores(patient_row, criterion)
                equal_similar_patients.extend(similar_patients[:num_per_criterion])

            # Ensure enough unique patients by adding more if needed
            if len(equal_similar_patients) < top_k:
                for criterion in criteria:
                    similar_patients = _extract_ids_and_scores(patient_row, criterion)
                    equal_similar_patients.extend(similar_patients[num_per_criterion:])
                    if len(equal_similar_patients) >= top_k:
                        break

            return _unique_top_k(equal_similar_patients, top_k)

        def _prioritized_by_criteria(criteria, patient_row, top_k):
            """
            Prioritizes similar patients by the first criterion, then fills with other criteria if necessary.
            """
            prioritized_similar_patients = []

            # Prioritize the first criterion
            primary_criterion = criteria[0]
            prioritized_similar_patients.extend(
                _extract_ids_and_scores(patient_row, primary_criterion)[:top_k]
            )

            # If not enough patients, add from other criteria
            if len(prioritized_similar_patients) < top_k:
                for criterion in criteria[1:]:
                    remaining = (top_k - len(prioritized_similar_patients)) // 2
                    prioritized_similar_patients.extend(
                        _extract_ids_and_scores(patient_row, criterion)[:remaining]
                    )
                    if len(prioritized_similar_patients) >= top_k:
                        break

            return _unique_top_k(prioritized_similar_patients, top_k)

        def _low_priority_ids(criteria, patient_row, top_k):
            """
            Selects the least prioritized patients, focusing on the top_k/2 of the second and third criteria.
            """
            low_priority_similar_patients = []
            half_k = max(1, top_k // 2)

            # Focus on the second and third criteria, taking the half_k from each
            for criterion in criteria[1:3]:
                similar_patients = _extract_ids_and_scores(patient_row, criterion)
                low_priority_similar_patients.extend(similar_patients[:half_k])

            return _unique_top_k(low_priority_similar_patients, top_k)

        if self.set_priority_of_similar_patients == "equal":
            similar_patients = _equal_priority(all_criteria, input_patient_row, input_top_k)
        elif self.set_priority_of_similar_patients == "prioritized":
            similar_patients = _prioritized_by_criteria(all_criteria, input_patient_row, input_top_k)
        elif self.set_priority_of_similar_patients == "low_priority":
            similar_patients = _low_priority_ids(all_criteria, input_patient_row, input_top_k)
        else:
            similar_patients = None
            print("Error: Not correct priority setting for similar patients")
        return similar_patients

    def load_similar_patients_mapping(self, top_k_data_path, criteria, hadm_id, top_k=6):
        if "score" in top_k_data_path.split("/")[-1]:
            # print(" NOTE: this similarity map has scores!")
            self._is_scored_similar_maps = True

        # patient_id is hadm_id here
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(top_k_data_path)

            # Check if patient_id exists in the DataFrame
            if hadm_id not in df['hadm_id'].values:
                raise ValueError(f"Patient with adm ID {hadm_id} not found in the dataset.")

            # Filter the DataFrame for the specified patient_id
            patient_row = df[df['hadm_id'] == hadm_id]

            # Get the top {top_k} most common similar patient IDs
            most_common_similar_patients = self.rerank_similar_patients(all_criteria=criteria,
                                                                        input_patient_row=patient_row,
                                                                        input_top_k=top_k)
            return most_common_similar_patients

        except Exception as e:
            print(f"Error loading similar patients for patient_id {hadm_id}: {e}")
            return []

    def generate_query_for_embedding_ranker(self, index, qa_file):
        """Generate a pure query (question + options + background), for embeddings scan method"""
        input_dict = self.placeholder
        init_query = "\n".join([f"{key}: '{{{value}}}" for key, value in input_dict.items()]) + "  question: {question}"
        simple_prompt_loader = QAPromptLoader(qa_file=qa_file)
        simple_prompt_loader.template = init_query
        simple_prompt_loader.current_id_index = index
        query = simple_prompt_loader.next_prompt(
            input_dict, update_index=False)
        return query

    def get_file_name_save_retrieved_content(self, hadm_id, top_k):
        
        priority = self.set_priority_of_similar_patients

        if self.rag_node_top_n == 6:
            str_rag_top_n = ""
        elif self.rag_node_top_n == 10:
            str_rag_top_n = "_"
        else:
            str_rag_top_n = f"_{str(self.rag_node_top_n)}"
        file_name = f"rag_r{str_rag_top_n}_{hadm_id}_top{top_k}_{self.dataset}_{priority}{EMBEDDING_SETTING}.txt"
        return file_name

    def get_or_copy_retrieved_file(
            self,
            patient_id: int,
            db_path: str,
            target_dir: str ,
    ) -> str:
        """
        Ensure 'retrieved_{patient_id}.txt' exists in
        target_folder/{target_patient_id}similar/ by:
          1. Looking there first.
          2. If missing, searching each subfolder of sister_folder
             and copying it in if found.
          3. Otherwise, running the SQL retrieval and saving it.

        Returns the absolute path to the file.
        """
        # Build the target directory and filename
        os.makedirs(target_dir, exist_ok=True)

        filename = f"retrieved_{patient_id}.txt"
        target_path = os.path.join(target_dir, filename)

        # 1. Check directly in the target folder
        if os.path.exists(target_path):
            print(f"Found existing file at {target_path}")
            return target_path

        # 2. Search in sister_folder's subdirectories
        sister_root = discharge_summary_dir + "similar_patients/"
        if os.path.isdir(sister_root):
            for sub in os.listdir(sister_root):
                subdir = os.path.join(sister_root, sub)
                candidate = os.path.join(subdir, filename)
                if os.path.exists(candidate):
                    shutil.copy2(candidate, target_path)
                    print(f"Copied {candidate} → {target_path}")
                    return target_path

        # 3. Fallback: retrieve from SQL and save
        sql = template_sql.replace("{patient_id}", str(patient_id))
        return save_sql_result(db_path, sql, str(patient_id), folder=target_dir)


    def _get_similar_folder(self, target_patient_id: str) -> str:
        """Return the absolute folder where this patient’s similar-notes live."""
        base = (
            f"../tmp/discharge_summary/similar_patients_embedding_rank{EMBEDDING_SETTING}/"
            if self.embedding_scan_ranker
            else "../tmp/discharge_summary/similar_patients/"
        )
        # note: dir_now is the directory of this script
        folder = os.path.join(dir_now, base, str(target_patient_id) + "similar")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _get_combined_paths(self, folder: str, target_patient_id: str, top_k: int):
        """Return (combined_notes_path, retrieved_content_path)."""
        combined_notes = os.path.join(
            folder, f"combined_{target_patient_id}_similar.txt"
        )
        retrieved_fname = self.get_file_name_save_retrieved_content(
            hadm_id=target_patient_id, top_k=top_k
        )
        retrieved_path = os.path.join(folder, retrieved_fname)
        return combined_notes, retrieved_path

    def _load_or_run_combined_rag(
            self,
            target_patient_id: str,
            qa_file: str,
            method: str,
            top_k: int,
    ) -> tuple[str, str]:
        """
        If a cached RAG output exists, load and return it;
        otherwise run a new RAG, save it, and return it.
        Returns: (rag_result, retrieved_content)
        """
        folder = self._get_similar_folder(target_patient_id)
        combined_notes, retrieved_path = self._get_combined_paths(
            folder, target_patient_id, top_k
        )

        if os.path.exists(retrieved_path):
            print(f"  Found saved RAG result at {retrieved_path}")
            with open(retrieved_path, "r", encoding="utf-8") as f:
                retrieved_content = f.read()
            # On reload, we typically want no new LLM call, so rag_result stays empty
            return "", retrieved_content

        # no cache → run new RAG
        print("Run new RAG")
        input_query = self.output_prompt(
            qa_file=qa_file,
            template_file=self.prompt_loader.template_file,
            knowledge_file=None,
            update_index=True,
        )
        rag_result, retrieved_content = self.output_rag_result(
            input_query, combined_notes, method=method
        )

        # write out the retrieved content for next time
        with open(retrieved_path, "w", encoding="utf-8") as f:
            f.write(retrieved_content)
        print(f"  Saved new RAG result to {retrieved_path}")
        print_gpu_utilization(f"Finish combined RAG for patient {target_patient_id}")

        return rag_result, retrieved_content

    def response_rag_summary(self, index, id_dict, *,
                             method="sentence",
                             is_similar_patients=False,
                             qa_file=None,
                             top_k=15,
                             top_k_data_path=...):
        target_patient_id = str(id_dict[index])

        # TODO GPT refined codes, need verify
        if is_similar_patients and self.set_rag_info_from_similar_patients == "combine":
            # handle the entire combine-branch up front, then return
            return self._load_or_run_combined_rag(
                target_patient_id, qa_file, method, top_k
            )

        if is_similar_patients:

            if self.embedding_scan_ranker:
                extra_folder = f"../tmp/discharge_summary/similar_patients_embedding_rank{EMBEDDING_SETTING}/"
            else:
                extra_folder = "../tmp/discharge_summary/similar_patients/"
            
            hadm_ids = [int(target_patient_id)]
            patient_id_list = []
            criteria = [f"top{self.similar_map_topk}_diagnoses", f"top{self.similar_map_topk}_procedures",
                        f"top{self.similar_map_topk}_prescriptions"]
            for hadm_id in hadm_ids:
                if self.embedding_scan_ranker:
                    list_path = os.path.join(script_dir,extra_folder, f"{target_patient_id}_similar_list_{self.dataset}_{top_k}.txt")

                    embedding_scan_results_csv_name = f"{target_patient_id}_embedding_scan_results_{self.dataset}_{top_k * 10}.csv"
                    embedding_scan_results_csv_path = os.path.join(script_dir,
                                                                   extra_folder,
                                                                   embedding_scan_results_csv_name)

                    if os.path.exists(list_path) and os.path.exists(embedding_scan_results_csv_path):
                        with open(list_path, "r", encoding="utf-8") as f:
                            similar_patients_adm_id_list = [
                                int(line.strip()) for line in f if line.strip()
                            ]
                        print(f"Loaded {len(similar_patients_adm_id_list)} IDs from: {list_path}")
                    else:
                        query_for_embedding_ranker = self.generate_query_for_embedding_ranker(index=index, qa_file=qa_file)
                        embedding_scan_results = search_query(query_for_embedding_ranker, embedding_dir=EMBEDDING_DIR, top_k=top_k*10)
                        similar_patients_adm_id_list = embedding_scan_results[0][0]

                        similar_patients_adm_id_list = [int(item.split(".")[0]) for item in similar_patients_adm_id_list if int(item.split(".")[0]) != int(target_patient_id)]
                        if len(similar_patients_adm_id_list) > top_k:
                            print(f"Find more than topk: {len(similar_patients_adm_id_list)}, chunk to {top_k}")
                            similar_patients_adm_id_list = similar_patients_adm_id_list[0:top_k]
                        with open(list_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(str(pid) for pid in similar_patients_adm_id_list))

                        self.save_id_scores_csv(results=embedding_scan_results,
                                                filepath=embedding_scan_results_csv_path)
                        print(f"Saved embedding scan results {embedding_scan_results_csv_name}")

                else: 
                    similar_patients_adm_id_list = self.load_similar_patients_mapping(
                        top_k_data_path,
                        criteria,
                        hadm_id=hadm_id,
                        top_k=top_k)
                patient_id_list.extend(similar_patients_adm_id_list)


        else:
            extra_folder = f"../tmp/discharge_summary/"
            update_next_prompt = True
            patient_id_list = [target_patient_id]

        overall_summary = " Useful info as reference from some similar patients (with ID): higher similarity sample first"

        print_gpu_utilization("Before main loop for similar patients")

        # Initiate knowledge for each different target patient
        self.prompt_loader.knowledge = "Useful info as reference from some similar patients (with ID): higher similarity sample first"

        similar_patients_notes_for_target_patient_folder = extra_folder+"/"+str(target_patient_id) + "similar/"

        for patient_id in patient_id_list:
            
            file_path = self.get_retrieved_file(
                patient_id=str(patient_id),
                target_folder=similar_patients_notes_for_target_patient_folder
            )

            if self.set_rag_info_from_similar_patients == "separate":
                print_gpu_utilization(f"Start patient  {patient_id}")

                # Load the summary from file to RAG file loader
                input_query_for_similar_patients = self.create_prompting_using_tmp_prompt_loader(qa_file, index, update_next_prompt)

                # summary: Get LLM returned summarized info from patient_id discharge summary
                try:
                    summary, _ = self.output_rag_result(input_query_for_similar_patients, file_path, method=method)
                except Exception as e:
                    print(f"ERROR on rag: {patient_id} {e}")
                    summary = ""
                print_gpu_utilization(f"Got summary  {patient_id}")
                extra_info_from_similar_patient = ""
                if summary != "":
                    extra_info_from_similar_patient = "\n (id " + str(patient_id) + "): " + summary + "\n"

                self.prompt_loader.knowledge = self.prompt_loader.knowledge + extra_info_from_similar_patient
                print_gpu_utilization(f"Got knowledge  {patient_id}")
                overall_summary = overall_summary + extra_info_from_similar_patient

                print_gpu_utilization(f"Finish patient  {patient_id}")
            else:
                # print("     Waiting for combining........")
                pass
        rag_result = overall_summary

        if self.set_rag_info_from_similar_patients == "combine":
            return self._load_or_run_combined_rag(
                target_patient_id, qa_file, method, top_k
            )

        # otherwise (separate mode), return what you accumulated
        return rag_result, ""


    def create_prompting_using_tmp_prompt_loader(self, qa_file, index, update_next_prompt):
        tmp_prompt_loader = QAPromptLoader(qa_file, None, None)
        tmp_prompt_loader.current_id_index = index
        tmp_prompt_loader.template_file = self.tmp_prompt_for_similar_patients
        tmp_prompt_loader.template = tmp_prompt_loader.load_template()
        question = tmp_prompt_loader.next_prompt(
            self.placeholder, update_next_prompt)
        # as retrieving info, question is updated from "what should be documented" to "what are documented"
        if self.set_rag_info_from_similar_patients == "separate":
            question = question.replace("should be", "are")
            question = question.replace("the best", "")
        return question

    def process_all_prompts(self,
                            qa_file=None, template_file=None, knowledge_file=None,
                            update_index=True, one_time_ask=True,
                            is_baseline=False,
                            is_rag_tool=False,
                            similarity_map_path=None,
                            rag_tool="sentence"):
        """Iterate all questions and have answers. """
        summary_dict = {}
        patient_id_dict = {}
        question_dict = {}

        if is_baseline:
            self.embedding_scan_ranker = True

        if is_rag_tool or not one_time_ask:
            self.extra_prompt_loader = QAPromptLoader(qa_file, None, None)
        if not self._is_qa_loaded:
            self.load_questions(qa_file, template_file, knowledge_file)
        if is_rag_tool or is_baseline:
            patient_id_dict = self.prompt_loader.get_info_dict("hadm_id")
            question_dict = self.prompt_loader.get_info_dict("question")
        result_dict = {}
        data_retrieved_dict = {}
        sql_codes_dict = {}
        db_path = os.getenv("DB_PATH")

        print_gpu_utilization("Before main loop for all patients")

        # Iterate all questions to have response
        index = 0
        while index < self.max_iteration:
            print(f"----Start index: {index}-------------------------------------------------------------------")
            retrieved_content = ""
            summary = ""
            if is_rag_tool:
                try:
                    summary, retrieved_content = self.response_rag_summary(index,
                                                       patient_id_dict,
                                                       method=rag_tool,
                                                       is_similar_patients=True,
                                                       qa_file=qa_file,
                                                       top_k_data_path=similarity_map_path)
                    if self.set_rag_info_from_similar_patients == "separate":
                        summary_dict.update({index: summary})
                        data_retrieved_dict = summary_dict
                except Exception as e:
                    print("!!!!!!!!!!!!!!!!!")
                    print(f"Error on retrieving rag: {e}")
                    print("!!!!!!!!!!!!!!!!!")
                    summary_dict.update({index:"Error"})
                    result_dict.update({index:"Error"})
                    print(" Finished question index: " + str(
                        index) + "-------------------------------------------------------------------" + "\n")
                    index += 1
                    continue
            if is_rag_tool and self.set_rag_info_from_similar_patients == "combine":
                
                if not self.llm_agent:
                    self.llm_agent = LLMAgent(llm=self.llm_version)
                # if a new RAG, retrieved_content is empty
                if retrieved_content == "":
                    one_response = summary
                # if there is already knowledge_file/RAG, retrieved_content is updated, so use that file
                else:
                    self.prompt_loader.knowledge = retrieved_content
                    one_prompt = self.output_prompt(qa_file, template_file, knowledge_file, update_index)
                    one_response = self.output_response(one_prompt).text
            else:
                one_prompt = self.output_prompt(qa_file, template_file, knowledge_file, update_index)
                one_response = self.output_response(one_prompt).text
            if not one_response:
                print("!!!!!!!!!!!")
                print("Error: No Response!")
                print("!!!!!!!!!!!")
            if one_time_ask:
                result_dict.update({index: one_response})

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(current_time + " Finished question index: " + str(index) + "-------------------------------------------------------------------" + "\n")
            index += 1
        return result_dict, data_retrieved_dict, sql_codes_dict

    def save_id_scores_csv(
            self,
            results: List[Tuple[List[str], List[float]]],
            filepath: str
    ) -> None:
        """
        Expects `results` like [( [id1.txt., id2, …], [score1, score2, …] ), …].
        Writes out a CSV with columns "id" and "score".
        """
        # take the first tuple (or loop over all if you have multiple)
        ids, scores = results[0]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score"])
            for _id_raw, score in zip(ids, scores):
                _id = _id_raw.split(".")[0]
                writer.writerow([_id, score])


    def extract_letter_from_result_dict(self, raw_result_dict, is_choice_only=False):
        def filter_keys_from_string(input_string):
            """
            Extracts the first characters (keys) from the input string and returns them concatenated as a single string.

            Example:
            Input: "L: Metoprolol Succinate, A: D5"
            Output: "LA"
            """
            # Split the input string by comma to get each key-value pair
            pairs = input_string.split(',')

            # Extract the first character (key) from each pair, strip any extra spaces
            keys = [pair.split(':')[0].strip() for pair in pairs]

            # Concatenate the keys and return them as a single string
            return ''.join(keys)

        answer_only_dict = {}
        reason_only_dict = {}
        for key in raw_result_dict.keys():
            answer_only_dict[key], reason_only_dict[key] = self.llm_agent.extract_answer_and_reason(
                raw_result_dict[key],
                dataset=self.dataset)

            if is_choice_only and answer_only_dict[key]:
                answer_only_dict[key] = filter_keys_from_string(answer_only_dict[key])

        return answer_only_dict, reason_only_dict

