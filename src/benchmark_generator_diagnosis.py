import os
from utils.pre_process.split_discharge_summary import DischargeSummarySplitter
from utils.prompt_utils import QAPromptLoader
from benchmark_generator import BenchmarkGenerator
import datetime
import yaml
import json

# Load Setup
dir_now = os.path.dirname(os.path.abspath(__file__))
config_dir = dir_now + "/../config/benchmark_generator/"
config_name = "config_benchmark_diagnoses_qa_generator.yaml"
with open(config_dir + config_name, "r") as file:
    config = yaml.safe_load(file)

# Access configuration values
llm = config['llm']
notes = config['notes']

dataset = config['dataset']
num_questions = config['num_questions']
prompt_file_generate_question = dir_now + config['paths']['prompt_file_generate_question']
note_database_file = dir_now + config['paths']['note_database_file']
mimic4_database_file = dir_now + config['paths']['mimic4_database_file']


class DischargeDiagnosisQAGenerator(BenchmarkGenerator):

    def __init__(self, llm="4o"):
        super().__init__(llm, template_file_path=prompt_file_generate_question)

    def get_discharge_diagnosis(self, note):
        # Split discharge summary and retrieve the Diagnosis part
        splitter = DischargeSummarySplitter(note, only_instruction=False, only_diagnosis=True)
        discharge_parts = splitter.process_summary()
        return discharge_parts.get("Discharge Diagnosis", [])

    def get_diagnoses_from_db(self, hadm_id, mimic4_database_file):
        # Query to get diagnosis data from the database
        query = f'''
        SELECT di.icd_code, d_icd.long_title
        FROM diagnoses_icd AS di
        JOIN d_icd_diagnoses AS d_icd
        ON di.icd_code = d_icd.icd_code 
        AND di.icd_version = d_icd.icd_version  -- Ensuring the ICD version matches
        WHERE di.hadm_id = {hadm_id};
        '''
        results = self.output_sql_result(db_path=mimic4_database_file, query=query)
        diagnoses_list = [f"{row[1]}" for row in results[0]]  # Extract long_title
        unique_diagnoses = set(diagnoses_list)
        return list(unique_diagnoses)

    def generate_qa_prompt(self, discharge_diagnosis="", diagnoses="", discharge_summary="", background=""):
        """Override the parent method to generate a QA prompt by replacing placeholders."""
        prompt_loader = QAPromptLoader(template_file=self.template_file_path)
        init_prompt = prompt_loader.load_template()

        # Replace the placeholders {discharge_diagnosis} and {diagnoses}
        prompt = prompt_loader.replace_place_holder(init_prompt, 'discharge_diagnosis', discharge_diagnosis)
        prompt = prompt_loader.replace_place_holder(prompt, 'diagnoses', diagnoses)
        if discharge_summary and background:
            prompt = prompt_loader.replace_place_holder(prompt, 'background', background)
            prompt = prompt_loader.replace_place_holder(prompt, 'discharge_summary', discharge_summary)

        return prompt

    def generate_qa_pairs(self, hadm_id, discharge_diagnosis, diagnoses, save_folder, discharge_summary_text):
        """Generate QA pairs using LLM."""
        # Step 0: Generate background
        other_part_splitter = DischargeSummarySplitter(discharge_summary_text, only_instruction=False)
        other_parts = other_part_splitter.process_summary()
        content_background = (str(other_parts.get("Presenting Condition")) +
                              str(other_parts.get("Clinical Assessment"))
                              )

        age_gender_string = self.generate_age_gender_string(db_path=mimic4_database_file,
                                                            hadm_id=hadm_id)
        basic_background = age_gender_string

        # Step 1: Generate prompt using the overridden function
        prompt = self.generate_qa_prompt(discharge_diagnosis=discharge_diagnosis,
                                         diagnoses=diagnoses,
                                         discharge_summary=content_background,
                                         background=basic_background)

        # Step 2: Use the prompt to feed into LLM
        llm_response = self.output_response_txt(prompt)

        # Step 3: Extract the JSON part from LLM response (assuming the LLM returns valid JSON)
        try:
            json_results = self.llm_agent.extract_code_blocks(input_text=llm_response, language='json')
            qa_pairs = json.loads(json_results[0] if json_results else llm_response)
        except Exception as e:
            print(f"Error extracting JSON from LLM response: {e}")
            return False

        # Step 4: Add hadm_id into the QA pair JSON
        qa_pairs['hadm_id'] = hadm_id
        qa_pairs['question'] = "Which diagnoses should be documented into the patient's discharge diagnosis?"

        # Step 6: Save the JSON to a file
        self.save_qa_to_file(qa_pairs, save_folder, hadm_id)
        return qa_pairs

    def generate_qa_for_hadm_id(self, hadm_id, note_database_file, mimic4_database_file, saved_note_folder, save_folder):
        # Step 1: Find the discharge note by hadm_id
        note = self.find_discharge_note_by_hadm_id(hadm_id, note_database_file, saved_note_folder)
        if not note:
            print(f"    Discharge note for HADM_ID {hadm_id} not found.")
            return False

        # Step 2: Get discharge diagnosis
        discharge_diagnosis = self.get_discharge_diagnosis(note)
        if "Not Found" in discharge_diagnosis or not discharge_diagnosis:
            print(f"    No discharge diagnosis found for HADM_ID {hadm_id}.")
            return False

        # Step 3: Get diagnoses from the database
        diagnoses = self.get_diagnoses_from_db(hadm_id, mimic4_database_file)
        if not diagnoses:
            print(f"    No diagnoses found for {hadm_id}")
            return False

        # Step 4: Generate QA pairs
        qa_pairs = self.generate_qa_pairs(hadm_id, discharge_diagnosis, diagnoses, save_folder, discharge_summary_text=note)

        if not qa_pairs:
            print(f"    No QA generated for {hadm_id}")
        else:
            self.append_to_csv(hadm_id,
                               qa_pairs,
                               save_folder + "/" + "history.csv",
                               info_name_from_note="discharge_diagnosis",
                               info_from_note=discharge_diagnosis,
                               info_name_from_table="diagnoses",
                               info_from_table=diagnoses
                               )


if __name__ == "__main__":
    qa_generator = DischargeDiagnosisQAGenerator()

    # from DataV2 similarity results; if not enough, look for ids in data/find_similar_patients/similar_patients_scores_0-2000_combined.csv
    with open(dir_now + '/../data/find_similar_patients/for_generating_diagnoses_QA_hadm_ids.txt', 'r') as f:
        list_hadm_id = [line.strip() for line in f]

    # Test setting
    list_hadm_id = list_hadm_id[0:50]

    print("The list of hadm_id to process: ")
    print(list_hadm_id)
    save_qa_to_diagnoses_main_folder = "/../data/generated_diagnoses_qa_json/"
    date_time_str_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_qa_to_folder = dir_now + save_qa_to_diagnoses_main_folder + date_time_str_folder
    if not os.path.exists(save_qa_to_folder):
        os.makedirs(save_qa_to_folder)

    folder_discharge_summary = dir_now + "/../tmp/discharge_summary/"

    id_count = 0
    for example_hadm_id in list_hadm_id:
        id_count = id_count + 1
        print(f"---------{id_count}-----------")
        print(f"Start generation: {example_hadm_id}")
        # Generate QA pairs for the given HADM_ID
        qa_generator.generate_qa_for_hadm_id(example_hadm_id,
                                             note_database_file,
                                             mimic4_database_file,
                                             saved_note_folder=folder_discharge_summary,
                                             save_folder=save_qa_to_folder)

        print(f"-------{id_count}-------------")
