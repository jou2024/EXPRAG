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
config_name = "config_benchmark_medications_qa_generator.yaml"
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


class DischargeMedicationQAGenerator(BenchmarkGenerator):

    def __init__(self, llm="4o"):
        super().__init__(llm, template_file_path=prompt_file_generate_question)

    def get_discharge_medications(self, note):
        # Split discharge summary and retrieve the Medications part
        splitter = DischargeSummarySplitter(note, only_instruction=False, only_medications=True)
        discharge_parts = splitter.process_summary()
        return discharge_parts.get("Discharge Medications", [])

    def generate_qa_prompt(self, discharge_medications="", prescriptions="", discharge_summary="", background=""):
        """Override the parent method to generate a QA prompt by replacing placeholders."""
        prompt_loader = QAPromptLoader(template_file=self.template_file_path)
        init_prompt = prompt_loader.load_template()

        # Replace the placeholders {discharge_medications} and {prescriptions}
        prompt = prompt_loader.replace_place_holder(init_prompt, 'discharge_medications', discharge_medications)
        prompt = prompt_loader.replace_place_holder(prompt, 'prescriptions', prescriptions)
        if discharge_summary and background:
            prompt = prompt_loader.replace_place_holder(prompt, 'background', background)
            prompt = prompt_loader.replace_place_holder(prompt, 'discharge_summary', discharge_summary)

        # print(prompt)
        return prompt

    def get_prescriptions_from_db(self, hadm_id, mimic4_database_file):
        # Query to get prescription data from the database
        query = f'''
        SELECT drug
        FROM prescriptions
        WHERE hadm_id = {hadm_id};
        '''
        query = query.replace("drug", "prod_strength")  # bug in the sql file
        results = self.output_sql_result(db_path=mimic4_database_file, query=query)
        # Get the unique prescription drugs
        list_prescriptions = [row[0] for row in results[0]]
        unique_prescriptions = set(list_prescriptions)
        # Check if the number of unique prescriptions is greater than 26
        if len(unique_prescriptions) > 52:
            print("   Too many prescriptions to create single-letter options: " + str(len(unique_prescriptions)))
            return False  # Too many prescriptions to create single-letter options
        return str(unique_prescriptions)

    def generate_qa_pairs(self, hadm_id, discharge_medications, prescriptions, save_folder, discharge_summary_text):
        """Generate QA pairs using LLM."""

        # Step 0: Generate background
        other_part_splitter = DischargeSummarySplitter(discharge_summary_text, only_instruction=False)
        other_parts = other_part_splitter.process_summary()
        content_background = (str(other_parts.get("Presenting Condition")) +
                              str(other_parts.get("In-Hospital Progress")) +
                              str(other_parts.get("Clinical Assessment"))
                              )
        age_gender_string = self.generate_age_gender_string(db_path=mimic4_database_file,
                                                            hadm_id=hadm_id)
        basic_background = age_gender_string

        # Step 1: Generate prompt using the overridden function
        prompt = self.generate_qa_prompt(discharge_medications=discharge_medications,
                                         prescriptions=prescriptions,
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
        qa_pairs['question'] ="Which medications should be prescribed to the patient at discharge?"

        # Step 6: Save the JSON to a file
        self.save_qa_to_file(qa_pairs, save_folder, hadm_id)
        return qa_pairs


    def generate_qa_for_hadm_id(self, hadm_id, note_database_file, mimic4_database_file, saved_note_folder, save_folder):
        # Step 1: Find the discharge note by hadm_id
        note = self.find_discharge_note_by_hadm_id(hadm_id, note_database_file, saved_note_folder)
        if not note:
            print(f"    Discharge note for HADM_ID {hadm_id} not found.")
            return False

        # Step 2: Get discharge medications
        discharge_medications = self.get_discharge_medications(note)
        if "Not Found" in discharge_medications or not discharge_medications:
            print(f"    No discharge medications found for HADM_ID {hadm_id}.")
            return False

        # Step 3: Get prescriptions from the database
        prescriptions = self.get_prescriptions_from_db(hadm_id, mimic4_database_file)
        if not prescriptions:
            print(f"    No prescriptions found for {hadm_id}")
            return False

        # Step 4: Generate QA pairs
        qa_pairs = self.generate_qa_pairs(hadm_id, discharge_medications, prescriptions, save_folder, discharge_summary_text=note)

        if not qa_pairs:
            print(f"    No QA generated for {hadm_id}")
        else:
            self.append_to_csv(hadm_id,
                               qa_pairs,
                               save_folder + "/" + "history.csv",
                               info_name_from_note="discharge_medications",
                               info_from_note=discharge_medications,
                               info_name_from_table="prescriptions",
                               info_from_table=prescriptions
                               )

if __name__ == "__main__":
    qa_generator = DischargeMedicationQAGenerator()

    # from DataV2 similarity results
    with open(dir_now + '/../data/find_similar_patients/for_generation_prescriptions_QA_hadm_ids.txt', 'r') as f:
        list_hadm_id = [line.strip() for line in f]

    # Test setting
    # list_hadm_id = ['27136630']
    list_hadm_id = list_hadm_id[0:150]

    print("The list of hadm_id to process: ")
    print(list_hadm_id)
    save_qa_to_medications_main_folder = "/../data/generated_medications_qa_json/"
    date_time_str_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_qa_to_folder = dir_now + save_qa_to_medications_main_folder + date_time_str_folder
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