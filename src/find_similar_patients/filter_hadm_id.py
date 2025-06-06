import pandas as pd
import json
import os
import numpy as np


dir_now = os.path.dirname(os.path.abspath(__file__))
dir_output_data = dir_now + '/../../data/find_similar_patients/'
dir_csv_source = dir_now + '/../../data/raw/mimic-iv-link/all_csv/'
dir_discharge_note_source = dir_now + '/../../data/raw/mimic-iv-link/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/'

class HadmIdProcessor:
    def __init__(self, admissions_path, diagnoses_path, procedures_path, prescriptions_path, discharge_path,
                 output_folder):
        self.admissions_path = admissions_path
        self.diagnoses_path = diagnoses_path
        self.procedures_path = procedures_path
        self.prescriptions_path = prescriptions_path
        self.discharge_path = discharge_path
        self.output_folder = output_folder

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

    def generate_live_hadm_id(self):
        live_hadm_id_path = os.path.join(self.output_folder, '1_live_hadm_id.json')
        if os.path.exists(live_hadm_id_path):
            print("Found 1_live_hadm_id, skip the first part")
            with open(live_hadm_id_path, 'r') as f:
                live_hadm_id_list = [np.int64(patient_id) for patient_id in json.load(f)]
            return live_hadm_id_list

        # Step 1: Read discharge.csv to get all hadm_ids
        if os.path.exists(self.discharge_path):
            print("Find csv discharge")
        discharge_df = pd.read_csv(self.discharge_path, on_bad_lines='skip')
        discharge_hadm_ids = set(discharge_df['hadm_id'].dropna().astype(int))

        # Step 2: Read admissions.csv and get hadm_ids with no "deathtime"
        admissions_df = pd.read_csv(self.admissions_path)
        live_admissions_df = admissions_df[pd.isna(admissions_df['deathtime'])]
        live_hadm_ids = set(live_admissions_df['hadm_id'].dropna().astype(int))

        # Step 3: Find the intersection of discharge_hadm_ids and live_hadm_ids
        live_hadm_id_list = list(discharge_hadm_ids & live_hadm_ids)
        print("Length of 1_live_hadm_id.json: ", len(live_hadm_id_list))

        discharge_df = None
        admissions_df = None

        # Step 4: Save the result as 1_live_hadm_id.json

        with open(live_hadm_id_path, 'w') as f:
            json.dump(live_hadm_id_list, f, indent=4)

        return live_hadm_id_list

    def generate_in_all_criteria(self, live_hadm_id_list):
        in_all_criteria_path = os.path.join(self.output_folder, '2_in_all_criteria.json')
        if os.path.exists(in_all_criteria_path):
            print("Found 2_in_all_criteria.json, skip the second step")
            with open(in_all_criteria_path, 'r') as f:
                in_all_criteria_hadm_ids = [np.int64(patient_id) for patient_id in json.load(f)]
            return in_all_criteria_hadm_ids

        # Step 1: Read diagnoses_icd.csv, procedures_icd.csv, prescriptions.csv
        print("Start to read diagnoses_df...")
        diagnoses_df = pd.read_csv(self.diagnoses_path)
        print("Start to read procedures_df...")
        procedures_df = pd.read_csv(self.procedures_path)
        print("Start to read prescriptions_df...")
        prescriptions_df = pd.read_csv(self.prescriptions_path)

        # Step 2: Extract unique hadm_id lists from each CSV file
        print("Have diagnoses_hadm_ids...")
        diagnoses_hadm_ids = set(diagnoses_df['hadm_id'].dropna().astype(int))
        diagnoses_df = None

        print("Have procedures_hadm_ids...")
        procedures_hadm_ids = set(procedures_df['hadm_id'].dropna().astype(int))
        procedures_df = None

        print("Have prescriptions_hadm_ids")
        prescriptions_hadm_ids = set(prescriptions_df['hadm_id'].dropna().astype(int))
        prescriptions_df = None
        live_hadm_ids = set(live_hadm_id_list)

        # Step 3: Find hadm_ids present in all 4 sets
        print("Start to calculate 2_in_all_criteria by intersection")
        in_all_criteria_hadm_ids = list(
            diagnoses_hadm_ids & procedures_hadm_ids & prescriptions_hadm_ids & live_hadm_ids)
        print("Length of 2_in_all_criteria.json: ", len(in_all_criteria_hadm_ids))

        # Step 4: Save the result as 2_in_all_criteria.json

        with open(in_all_criteria_path, 'w') as f:
            json.dump(in_all_criteria_hadm_ids, f, indent=4)

        return in_all_criteria_hadm_ids

    def filter_ids_by_item_count(self, input_ids, csv_path, output_file_name, min_count=3, max_count=40):
        """
        Filters hadm_ids based on the count of items associated with them in a CSV file.

        :param input_ids: List of hadm_ids to filter
        :param csv_path: Path to the CSV file to check (diagnoses, procedures, or prescriptions)
        :param output_file_name: Name of the output JSON file
        :param min_count: Minimum number of items required for each hadm_id
        :param max_count: Maximum number of items allowed for each hadm_id
        :return: Filtered list of hadm_ids meeting the criteria
        """
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Filter for hadm_ids in input_ids and count occurrences
        filtered_ids = []
        for hadm_id in input_ids:
            count = df[df['hadm_id'] == hadm_id].shape[0]
            if min_count <= count <= max_count:
                filtered_ids.append(hadm_id)

        # Save the filtered list to a JSON file
        output_path = os.path.join(self.output_folder, output_file_name)
        with open(output_path, 'w') as f:
            json.dump(filtered_ids, f, indent=4)

        return filtered_ids


    def process(self):
        # Generate the first output
        live_hadm_id_list = self.generate_live_hadm_id()

        # Generate the second output
        in_all_criteria_hadm_ids = self.generate_in_all_criteria(live_hadm_id_list)

        # Step 3: Filter by item count in diagnoses_icd.csv
        print("Start to run and count ids in diagnoses")
        diagnoses_filtered_ids = self.filter_ids_by_item_count(
            in_all_criteria_hadm_ids, self.diagnoses_path, '3_3-40_diagnoses.json'
        )

        # Step 4: Filter by item count in procedures_icd.csv using results from diagnoses filter
        print("Start to run and count ids in procedures")
        procedures_filtered_ids = self.filter_ids_by_item_count(
            diagnoses_filtered_ids, self.procedures_path, '4_3-40_procedures.json'
        )

        # Step 5: Filter by item count in prescriptions.csv using results from procedures filter
        print("Start to run and count ids in prescriptions")
        self.filter_ids_by_item_count(
            procedures_filtered_ids, self.prescriptions_path, '5_3-40_prescriptions.json'
        )


# Example usage
output_folder = dir_output_data
processor = HadmIdProcessor(
    admissions_path=dir_csv_source+'admissions.csv',
    diagnoses_path=dir_csv_source+'diagnoses_icd.csv',
    procedures_path=dir_csv_source+'procedures_icd.csv',
    prescriptions_path=dir_csv_source+'prescriptions.csv',
    discharge_path=dir_discharge_note_source+'discharge.csv',
    output_folder=output_folder
)
processor.process()
