import os
import datetime
import pandas as pd
import yaml
import json

from utils.llm_utils import LLMAgent
from utils.prompt_utils import QAPromptLoader
from utils.sql_utils import SQLiteDB,save_sql_result

from utils.pre_process.split_discharge_summary import DischargeSummarySplitter
from utils.pre_process.organize_json_points import process_points


# Load Setup
dir_now = os.path.dirname(os.path.abspath(__file__))
config_dir = dir_now + "/../config/benchmark_generator/"
config_name = "config_benchmark_generator.yaml"
with open(config_dir + config_name, "r") as file:
    config = yaml.safe_load(file)

# Access configuration values
llm = config['llm']
notes = config['notes']

dataset = config['dataset']
num_questions = config['num_questions']

note_database_file = dir_now + config['paths']['note_database_file']
mimic4_database_file = dir_now + config['paths']['mimic4_database_file']
prompt_file_generate_question = dir_now + config['paths']['prompt_file_generate_question']
prompt_file_filter_context = dir_now + config['paths']['prompt_file_filter_context']

class BenchmarkGenerator:
    def __init__(self, llm="4o", template_file_path=""):
        self.llm_agent = None
        self.llm_version = llm
        self.prompt_loader = None
        self.sql_operator = None
        self.database = None
        self.num_questions = 0

        self.template_file_path = template_file_path

    def load_database(self, db_path):
        self.sql_operator = SQLiteDB
        self.database = self.sql_operator(db_path)

    def set_max_iteration(self, num_questions):
        self.num_questions = num_questions

    def output_response_txt(self, prompt):
        self.llm_agent = LLMAgent(llm=self.llm_version)
        print(" Send prompt and wait for response...")
        response = self.llm_agent.send_msg(prompt)
        print(" Got response")
        return response.text

    def output_sql_result(self, db_path="", query=""):
        self.load_database(db_path)
        db = self.database
        db.connect()
        results = db.execute_query(query)

        # print("Result from SQL: ")
        # for row in results:
        #     print(row)

        db.close()
        return results

    def fetch_basic_info_from_db(self, db_path, hadm_id=0):
        if hadm_id ==0:
            print("Miss hadmi_id")
            return False
        sql_get_subject_id = f'''
        SELECT admittime, subject_id 
        FROM admissions 
        WHERE hadm_id = {hadm_id};
        '''
        admission_data = self.output_sql_result(db_path, sql_get_subject_id)[0][0]
        admittime, subject_id = admission_data
        sql_get_info = f'''
        SELECT gender, anchor_age, anchor_year 
        FROM patients 
        WHERE subject_id = {subject_id};
        '''
        patient_data = self.output_sql_result(db_path, sql_get_info)[0][0]
        return admittime, patient_data

    def generate_age_gender_string(self, db_path, hadm_id=0):
        admittime, patient_data = self.fetch_basic_info_from_db(db_path, hadm_id)
        gender, anchor_age, anchor_year = patient_data
        admittime_dt = datetime.datetime.strptime(admittime, "%Y-%m-%d %H:%M:%S")
        admit_year = admittime_dt.year
        admi_age = (admit_year - anchor_year) + anchor_age
        gender_str = "male" if gender == "M" else "female"
        output_str = f"a {admi_age} years old {gender_str}"
        return output_str

    def generate_qa_prompt(self, discharge_summary="", hadm_id="", background="", points_of_instructions="", points_to_avoid=""):
        prompt_loader = QAPromptLoader(template_file=self.template_file_path)
        init_prompt = prompt_loader.load_template()
        if discharge_summary and background and hadm_id:
            add_info_prompt = prompt_loader.replace_place_holder(init_prompt, 'discharge_summary', discharge_summary)
            add_info_prompt = prompt_loader.replace_place_holder(add_info_prompt, 'points_of_instructions', points_of_instructions)
            add_info_prompt = prompt_loader.replace_place_holder(add_info_prompt, 'points_to_avoid', points_to_avoid)
            add_info_prompt = prompt_loader.replace_place_holder(add_info_prompt, 'hadm_id', hadm_id)
            final_prompt = prompt_loader.replace_place_holder(add_info_prompt, 'background', background)
            return final_prompt
        else:
            print("Not enough info from discharge_summary hadm_id background")
            return False

    def generate_filter_prompt(self, prompt_template_path="", context=""):
        prompt_loader = QAPromptLoader(template_file=prompt_template_path)
        init_prompt = prompt_loader.load_template()
        if context:
            add_discharge_context_prompt = prompt_loader.replace_place_holder(init_prompt, 'context', context)
            return add_discharge_context_prompt
        else:
            print("Not enough info from context")
            return False

    def organize_instruction_points(self, content):
        json_content = json.loads(content)
        organized_instruction_points = process_points(json_content)
        return organized_instruction_points

    def find_discharge_note_by_hadm_id(self, hadm_id, note_database_file, folder_path):
        # Construct the expected file name
        file_name = f"retrieved_{hadm_id}.txt"

        # Search through the folder and its subfolders for the file
        for root, dirs, files in os.walk(folder_path):
            if file_name in files:
                file_path = os.path.join(root, file_name)
                print(f"    Found retrieved note for HADM_ID {hadm_id} in {file_path}.")
                # Return the content of the retrieved file
                with open(file_path, 'r') as file:
                    return file.read()

        # If the file does not exist, execute the SQL query to retrieve the note
        print(f"    Note for HADM_ID {hadm_id} not found in folder, fetching from database.")
        query = f'''
        SELECT text
        FROM discharge
        WHERE hadm_id = {hadm_id}
        LIMIT 5;
        '''
        discharge_summary_file_path = save_sql_result(db_path=note_database_file,
                                                      query=query,
                                                      index=hadm_id,
                                                      folder="/../tmp/discharge_summary/")
        with open(discharge_summary_file_path, 'r') as discharge_summary_file:
            print(f"    Note for HADM_ID {hadm_id} saved to {discharge_summary_file_path}.")
            return discharge_summary_file.read()

    def save_qa_to_file(self, qa_pairs, save_folder, hadm_id):
        """Save the generated QA pairs to a JSON file."""
        file_path = os.path.join(save_folder, f"{hadm_id}_qa_pairs.json")
        with open(file_path, "w") as file:
            json.dump(qa_pairs, file, indent=4)
        print(f"QA pairs saved to {file_path}")

    def append_to_csv(self,
                      hadm_id,
                      qa_pairs,
                      output_csv_path,
                      info_name_from_note="",
                      info_from_note=None,
                      info_name_from_table="",
                      info_from_table=None
                      ):
        # Create a dictionary with the input values
        new_data = {
            'hadm_id': [str(hadm_id)],
            info_name_from_note: [str(info_from_note)],
            info_name_from_table: [str(info_from_table)],
            'qa_pairs': [str(qa_pairs)]
        }

        # Convert the dictionary to a DataFrame
        df_new = pd.DataFrame(new_data)

        # Check if the CSV file exists
        if not os.path.exists(output_csv_path):
            # If the file does not exist, create it and write the header
            df_new.to_csv(output_csv_path, index=False, mode='w')
        else:
            # If the file exists, append the new data without writing the header again
            df_new.to_csv(output_csv_path, index=False, mode='a', header=False)

        print(f"Appended data to {output_csv_path}")


if __name__ == "__main__":
    print("Instruction Generator starts...")
    print("")
    instruction_qa_generator = BenchmarkGenerator(template_file_path=prompt_file_generate_question)

    # Different groups for testing filter and splitter
    # 1 Empty result: need spliter update
    # Note '25433541': actual no instruction
    # Note '22557235':
    # list_hadm_id = ['22557235', '22036908']

    # from data v2
    with open(dir_now + '/../data/test/similar_hadm_id_statistic_2024-11-10_00-06/top_30_percent_top30_diagnoses_hadm_ids.txt', 'r') as f:
        list_hadm_id = [line.strip() for line in f]

    # Test setting
    # list_hadm_id = ['20531549']
    list_hadm_id = list_hadm_id[150:210]

    print("The list of hadm_id to process: ")
    print(list_hadm_id)
    save_qa_to_qa_folder = "/../data/generated_qa_json/"
    folder = "/../tmp/discharge_summary/"
    date_time_str_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_qa_to_folder = dir_now + save_qa_to_qa_folder + date_time_str_folder
    if not os.path.exists(save_qa_to_folder):
        os.makedirs(save_qa_to_folder)

    id_count = 0
    qa_count = 0
    for example_hadm_id in list_hadm_id:
        id_count = id_count + 1
        print(f"---------{id_count}-----------")
        print(f"Start generation: {example_hadm_id}")
        # 2. Prepare patient background info for prompt
        #     1. from table patient:
        #         1. find age
        #         2. find gender
        age_gender_string = instruction_qa_generator.generate_age_gender_string(db_path=mimic4_database_file, hadm_id=example_hadm_id)
        example_background = age_gender_string

        # 3. Prepare correct section of discharge summary
        # check and load txt file, or find raw discharge summary from hadm_id, save to file
        # split the file, output the last portion
        # 3.1  SQL Template
        template_sql = '''SELECT text
                        FROM discharge
                        where hadm_id = {patient_id}
                        LIMIT 10;'''

        # 3.2 Save text of Discharge Summary to file
        discharge_summary_file_path = dir_now + folder + f"retrieved_{example_hadm_id}.txt"
        if not os.path.exists(discharge_summary_file_path):
            updated_sql = template_sql.replace("{patient_id}", example_hadm_id)
            discharge_summary_file_path = save_sql_result(db_path=note_database_file,
                            query=updated_sql,
                            index=example_hadm_id,
                            folder=folder)
        else:
            print(" Found retrieved data from: " + discharge_summary_file_path)
            print(" No need to retrieve again")

        with open(discharge_summary_file_path, 'r') as file:
            discharge_summary_text = file.read()
        other_part_splitter = DischargeSummarySplitter(discharge_summary_text, only_instruction=False)
        other_parts = other_part_splitter.process_summary()
        example_content_background = (str(other_parts.get("Presenting Condition")) +
                                      str(other_parts.get("In-Hospital Progress")))

        # 3.3 Split and select only instruction parts
        instruction_splitter = DischargeSummarySplitter(discharge_summary_text, only_instruction=True)
        instruction_parts = instruction_splitter.process_summary()
        # Only 2 portion "Discharge Instructions" or/and "Followup Instructions"
        # Example from below:
        example_context_discharge_instruction = instruction_parts.get("Discharge Summary")

        date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_content_file_name = f"{example_hadm_id}_filtered_content_{date_time_str}.json"
        save_content_file_path = save_qa_to_folder + "/" + save_content_file_name

        # 4.1 Generate prompt for filter
        filter_prompt = instruction_qa_generator.generate_filter_prompt(prompt_template_path=prompt_file_filter_context,
                                                         context=example_context_discharge_instruction)
        # 4.2 Have filtered content and points
        filter_response_text = instruction_qa_generator.output_response_txt(filter_prompt)
        try:
            content_json_results = instruction_qa_generator.llm_agent.extract_code_blocks(input_text=filter_response_text, language='json')
            content_json_result = content_json_results[0]
            content_result = content_json_result
            # print(json_result)
        except:
            print(" Cannot find json_results")
            print(" Save the raw result ")
            # print(qa_response_text)
            content_result = filter_response_text

        with open(save_content_file_path, "x") as file:
            file.write(content_result)
            print(" Saved filtered content and points to: " + save_content_file_name)

        # 4.3 Load points
        example_instructions_list = instruction_qa_generator.organize_instruction_points(content_result)
        if not example_instructions_list:
            print("Not enough points for id: " + str(example_hadm_id))
            print("Go to next ID")
            print(f"--------{id_count}-------------")
            continue

        for example_instructions_one_question in example_instructions_list:
            other_instructions_points = example_instructions_list.copy()
            other_instructions_points.remove(example_instructions_one_question)
        
            # 5. Generate prompt for questions
            qa_prompt = instruction_qa_generator.generate_qa_prompt(
                                               discharge_summary=example_content_background,
                                               points_of_instructions=example_instructions_one_question,
                                               points_to_avoid=other_instructions_points,
                                               hadm_id=example_hadm_id,
                                               background=example_background
                                               )
            # print(qa_prompt)

            # 6. Generate QA
            qa_response_text = instruction_qa_generator.output_response_txt(qa_prompt)
            date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_qa_file_name = f"{example_hadm_id}_generated_qa_{date_time_str}.json"

            save_qa_file_path = save_qa_to_folder + "/" + save_qa_file_name
            try:
                json_results = instruction_qa_generator.llm_agent.extract_code_blocks(input_text=qa_response_text, language='json')
                json_result = json_results[0]
                qa_result = json_result
                # print(json_result)
            except:
                print(" Cannot find json_results")
                print(" Save the raw result ")
                # print(qa_response_text)
                qa_result = qa_response_text

            qa_count = qa_count + 1
            with open(save_qa_file_path, "x") as file:
                file.write(qa_result)
                print(" Saved generated qa to: " + save_qa_file_name)
            print(f"----qa----{qa_count}")
        print(f"-------{id_count}-------------")


