import sqlite3
import pandas as pd

import logging
from collections import defaultdict, Counter
from math import sqrt, isnan, ceil

import os
import datetime
import json
import numpy as np

import random

# Get the current date and time
now = datetime.datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

dir_now = os.path.dirname(os.path.abspath(__file__))
dir_log = dir_now + '/../../data/logs/'

# Setting up logging
is_log_time = False
if is_log_time:
    logging.basicConfig(filename=dir_log + 'similarity_log' + date_time_str + '.txt', level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(message)s')
else:
    logging.basicConfig(filename=dir_log+'similarity_log'+date_time_str+'.txt', level=logging.INFO,
                        format='%(levelname)s - %(message)s')


class EHRSimilarity:
    def __init__(self,
        db_path,
        test_mode=False,
        top_n=10,
        log_on=True,
        csv_mode=False,
        score_mode=False,
        split_bins=False  # new flag
    ):
        self.db_path = db_path
        self.test_mode = test_mode
        self.log_on = log_on
        self.with_score = score_mode

        self.csv_mode = csv_mode
        self.split_bins = split_bins

        if csv_mode:
            logging.info(f"Read CSV Starts")
            print(f"Read CSV Starts")
            self.admissions = pd.read_csv(os.path.join(db_path, 'admissions.csv'))
            self.diagnoses_icd = pd.read_csv(os.path.join(db_path, 'diagnoses_icd.csv'))
            self.procedures_icd = pd.read_csv(os.path.join(db_path, 'procedures_icd.csv'))
            self.prescriptions = pd.read_csv(os.path.join(db_path, 'prescriptions.csv'))
            logging.info(f"Read CSV Done")
            print(f"Read CSV Done")
        else:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

        self.similar_patients = defaultdict(dict)

        self.top_n = top_n
        self.top_n_diagnoses = 'top' + str(top_n) + '_diagnoses'
        self.top_n_procedures = 'top' + str(top_n) + '_procedures'
        self.top_n_prescriptions = 'top' + str(top_n) + '_prescriptions'

    # For subject_id
    def get_icd_codes(self, table, subject_id, info="icd_code"):

        if self.csv_mode:
            df = table
            return set(df[df['subject_id'] == subject_id][info])

        query = f"SELECT {info} FROM {table} WHERE subject_id = ?"
        self.cursor.execute(query, (subject_id,))
        return set(row[0] for row in self.cursor.fetchall())

    # For hadm_id
    def get_icd_codes_by_hadm(self, table, hadm_id, info="icd_code"):
        """
        Fetch ICD codes based on hadm_id.
        """
        if self.csv_mode:
            df = table
            # print("Start to load csv")
            code = set(df[df['hadm_id'] == hadm_id][info])
            # print("End to load csv")
            return code
        else:
            query = f"SELECT {info} FROM {table} WHERE hadm_id = ?"
            self.cursor.execute(query, (hadm_id,))
            return set(row[0] for row in self.cursor.fetchall())

    def fetch_all_subject_id(self):
        if self.csv_mode:
            return self.admissions['subject_id'].unique()

        else:
            self.cursor.execute("SELECT DISTINCT subject_id FROM admissions")
            subject_ids = [row[0] for row in self.cursor.fetchall()]
            return subject_ids

    def fetch_all_hadm_id(self):
        if self.csv_mode:
            return self.admissions['hadm_id'].unique()

        else:
            self.cursor.execute("SELECT DISTINCT hadm_id FROM admissions")
            subject_ids = [row[0] for row in self.cursor.fetchall()]
            return subject_ids

    def fetch_hadm_ids_by_subject_id(self, subject_ids):
        """
        Fetch all hadm_ids for the given list of subject_ids.
        """
        if self.csv_mode:
            df = self.diagnoses_icd  # Assuming all tables have the hadm_id info
            hadm_ids = df[df['subject_id'].isin(subject_ids)]['hadm_id'].unique()
        else:
            query = f"SELECT DISTINCT hadm_id FROM admissions WHERE subject_id IN ({','.join('?' for _ in subject_ids)})"
            self.cursor.execute(query, subject_ids)
            hadm_ids = [row[0] for row in self.cursor.fetchall()]
        return hadm_ids

    # Method 1 to calculate similarity: not a good way
    def cosine_similarity(self, list1, list2):
        vec1, vec2 = Counter(list1), Counter(list2)
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = sqrt(sum1) * sqrt(sum2)

        return numerator / denominator if denominator != 0 else 0

    # Method 2 to calculate similarity: preferred
    def jaccard_similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = set1 & set2
        union = set1 | set2
        similarity_result = len(intersection) / len(union) if len(union) != 0 else 0.0
        rounded_similarity_result = round(similarity_result, 4)
        return rounded_similarity_result

    def filter_by_length(self, set1, set2, threshold=10):
        """
        Filter out sets based on significant length differences or empty/nan sets.
        """
        # threshold 10
        len1, len2 = len(set1), len(set2)
        intersection = len(set(set1) & set(set2))

        # If one set is significantly longer than other but the intersection is not proportionally large, filter out
        if len1 * threshold < len2 and intersection < len1:
            return False
        if len2 * threshold < len1 and intersection < len2:
            return False

        # Filter out empty inputs
        if any(length == 0 for length in (len1, len2)):
            return False

        # Filter out "nan" inputs
        if any(length == 1 for length in (len1, len2)):
            only_element_1 = next(iter(set1))
            only_element_2 = next(iter(set2))
            if isinstance(only_element_1, float):
                if isnan(only_element_1):
                    return False
            if isinstance(only_element_2, float):
                if isnan(only_element_2):
                    return False

        return True

    def calculate_similarity(self, patient_a, patient_b):
        similarities = {"shared_diagnoses": 0.0,
                        "shared_procedures": 0.0,
                        "shared_prescriptions": 0.0
                        }

        # Use a loop to apply same way to all elements/criteria
        elements = ['diagnoses', 'procedures', 'prescriptions']

        for element in elements:
            set1, set2 = patient_a[element], patient_b[element]
            if self.filter_by_length(set1, set2, threshold=10):
                similarities[f'shared_{element}'] = self.jaccard_similarity(set1, set2)
            else:
                similarities[f'shared_{element}'] = 0.0

        # for key, value in similarities.items():
        #     if value == 1:
        #         print(f"100% same of variable Name: {key}")

        return similarities['shared_diagnoses'], similarities['shared_procedures'], similarities['shared_prescriptions']

    def find_top_similar_hadm_ids(self, hadm_ids, single_hadm_id=None):
        """
        Calculate similarities based on hadm_id instead of subject_id.
        hadm_ids: all reference hospital admission id as similarity candidates
        single_hadm_id: if true: target hospital admission id(s) to investigate
        """
        if self.csv_mode:
            diagnoses_icd = self.diagnoses_icd
            procedures_icd = self.procedures_icd
            prescriptions = self.prescriptions
        else:
            diagnoses_icd = 'diagnoses_icd'
            procedures_icd = 'procedures_icd'
            prescriptions = 'prescriptions'

        top_n = self.top_n
        admissions = {}

        logging.info(f"Fetch all info Starts")

        # Load patient admission data
        for i, hadm_id in enumerate(hadm_ids):
            admissions[hadm_id] = {
                'diagnoses': self.get_icd_codes_by_hadm(diagnoses_icd, hadm_id),
                'procedures': self.get_icd_codes_by_hadm(procedures_icd, hadm_id),
                'prescriptions': self.get_icd_codes_by_hadm(prescriptions, hadm_id, info="formulary_drug_cd")
            }
            if i % 1000 == 0:
                print(i, "adm id loaded")

        logging.info("Fetch all info Ends")

        # Determine the loop targets
        loop_hadm_ids = single_hadm_id if single_hadm_id is not None else hadm_ids

        for i, hadm_id in enumerate(loop_hadm_ids):
            try:
                if self.test_mode or self.log_on:
                    logging.info(f"{hadm_id} Starts:")
                    print(i)
                    print(f"  {hadm_id} ")

                similarities = []

                # self loop for testing
                # for other_hadm_id in loop_hadm_ids:

                # Real loop for all candidates
                for other_hadm_id in hadm_ids:
                    if hadm_id != other_hadm_id:
                        try:
                            shared_diagnoses, shared_procedures, shared_prescriptions = self.calculate_similarity(
                                admissions[hadm_id], admissions[other_hadm_id])

                            # Filter out if all criteria are 0
                            if shared_diagnoses != 0.0 or shared_procedures != 0.0 or shared_prescriptions != 0.0:
                                if (shared_diagnoses + shared_procedures + shared_prescriptions >= 0.5) or self.split_bins:
                                    if self.test_mode or self.log_on:
                                        logging.info(
                                            f"  vs {other_hadm_id} {str(shared_diagnoses)} , {str(shared_procedures)} , {str(shared_prescriptions)} ")
                                    data_tuple = (other_hadm_id, shared_diagnoses, shared_procedures, shared_prescriptions)
                                    if len(data_tuple) == 4:
                                        similarities.append(data_tuple)

                        except Exception as e:
                            logging.error(
                                f"Error on similarity check of hadm_id {hadm_id} and other_hadm_id {other_hadm_id}: {e}")
                            print(
                                f"Error on similarity check of hadm_id {hadm_id} and other_hadm_id {other_hadm_id}: {e}")

                if self.split_bins and self.with_score:
                    # split full lists into quintile bins and select top_n per bin
                    def split_and_select(lst):
                        m = len(lst)
                        size = ceil(m / 5)
                        bins = []
                        for i in range(5):
                            start = i * size
                            end = m if i == 4 else (i + 1) * size
                            bins.append(lst[start:end])
                        return bins

                    # for each criterion, prepare sorted list
                    diag_list = sorted([(x[0], x[1]) for x in similarities], key=lambda x: -x[1])
                    proc_list = sorted([(x[0], x[2]) for x in similarities], key=lambda y: -y[1])
                    pres_list = sorted([(x[0], x[3]) for x in similarities], key=lambda z: -z[1])

                    # split and assign
                    for i, (lst, key_prefix) in enumerate([
                        (diag_list, self.top_n_diagnoses),
                        (proc_list, self.top_n_procedures),
                        (pres_list, self.top_n_prescriptions)
                    ]):
                        bins = split_and_select(lst)
                        for j, bin_list in enumerate(bins):
                            key = f"{key_prefix}_{j * 20}_{(j + 1) * 20}"
                            # select top_n from this bin
                            self.similar_patients[hadm_id][key] = [f"{pid}:{score}" for pid, score in
                                                                   bin_list[:self.top_n]]
                # Store similarities with or without scores
                elif self.with_score:
                    self.similar_patients[hadm_id][self.top_n_diagnoses] = [
                                                                               f"{x[0]}:{x[1]}" for x in
                                                                               sorted(similarities,
                                                                                      key=lambda x: -x[1]) if
                                                                               x[1] > 0.0][:top_n]

                    self.similar_patients[hadm_id][self.top_n_procedures] = [
                                                                                f"{x[0]}:{x[2]}" for x in
                                                                                sorted(similarities,
                                                                                       key=lambda x: -x[2]) if
                                                                                x[2] > 0.0][:top_n]

                    self.similar_patients[hadm_id][self.top_n_prescriptions] = [
                                                                                   f"{x[0]}:{x[3]}" for x in
                                                                                   sorted(similarities,
                                                                                          key=lambda x: -x[3]) if
                                                                                   x[3] > 0.0][:top_n]
                else:
                    self.similar_patients[hadm_id][self.top_n_diagnoses] = [
                                                                               x[0] for x in sorted(similarities,
                                                                                                    key=lambda x: -
                                                                                                    x[1]) if
                                                                               x[1] > 0.0][:top_n]

                    self.similar_patients[hadm_id][self.top_n_procedures] = [
                                                                                x[0] for x in sorted(similarities,
                                                                                                     key=lambda x: -
                                                                                                     x[2]) if
                                                                                x[2] > 0.0][:top_n]

                    self.similar_patients[hadm_id][self.top_n_prescriptions] = [
                                                                                   x[0] for x in
                                                                                   sorted(similarities,
                                                                                          key=lambda x: -x[3]) if
                                                                                   x[3] > 0.0][:top_n]

                logging.info(f"{hadm_id} as hadm_id loop ends")

            except Exception as e:
                logging.error(f"Error processing hadm_id {hadm_id}: {e}")
                print(f"Error processing hadm_id {hadm_id}: {e}")


    def save_to_csv_bins(self, note=""):
        """
        Save the quintile-split results to CSV. Requires split_bins=True and score_mode=True.
        """
        records = []
        for patient_id, data in self.similar_patients.items():
            record = {'hadm_id': patient_id}
            # iterate through each criterion and its bins
            for key_prefix in [self.top_n_diagnoses, self.top_n_procedures, self.top_n_prescriptions]:
                for i in range(5):
                    key = f"{key_prefix}_{i*20}_{(i+1)*20}"
                    record[key] = ','.join(data.get(key, []))
            records.append(record)

        df = pd.DataFrame(records)
        out_dir = os.path.join('..', '..', 'data', 'find_similar_patients')
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(
            os.path.join(out_dir, f'similar_patients_bins_{note}_{date_time_str}.csv'),
            index=False
        )

    def save_to_db(self):
        """
        Save result to db file for SQL. Not for csv mode
        """
        try:
            self.cursor.execute(
                f"CREATE TABLE IF NOT EXISTS similar_patients (subject_id INTEGER PRIMARY KEY, {self.top_n_diagnoses} TEXT, {self.top_n_procedures} TEXT, {self.top_n_prescriptions} TEXT)")
            for patient_id, data in self.similar_patients.items():
                # Define the SQL query with placeholders for the column names and values
                sql = """
                INSERT OR REPLACE INTO similar_patients (subject_id, {0}, {1}, {2})
                VALUES (?, ?, ?, ?)
                """.format(self.top_n_diagnoses, self.top_n_procedures, self.top_n_prescriptions)

                # Data to be inserted
                values = (
                    patient_id,
                    ','.join(map(str, data[self.top_n_diagnoses])),
                    ','.join(map(str, data[self.top_n_procedures])),
                    ','.join(map(str, data[self.top_n_prescriptions]))
                )

                self.cursor.execute(sql, values)
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error saving to database: {e}")

    def close(self):
        self.conn.close()

    def save_to_csv(self, note=""):
        try:
            result = []
            for patient_id, data in self.similar_patients.items():
                result.append({
                    'hadm_id': patient_id,
                    self.top_n_diagnoses: ','.join(map(str, data[self.top_n_diagnoses])),
                    self.top_n_procedures: ','.join(map(str, data[self.top_n_procedures])),
                    self.top_n_prescriptions: ','.join(map(str, data[self.top_n_prescriptions]))
                })
            result_df = pd.DataFrame(result)
            result_df.to_csv(os.path.join(self.db_path, f'similar_patients_{note}.csv'), index=False)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")

    def save_to_csv_with_scores(self, note=""):
        try:
            result = []
            for patient_id, data in self.similar_patients.items():
                result.append({
                    'hadm_id': patient_id,
                    self.top_n_diagnoses: ','.join(map(str, data[self.top_n_diagnoses])),
                    self.top_n_procedures: ','.join(map(str, data[self.top_n_procedures])),
                    self.top_n_prescriptions: ','.join(map(str, data[self.top_n_prescriptions]))
                })
            result_df = pd.DataFrame(result)
            result_df.to_csv(os.path.join(dir_now + '/../../data/find_similar_patients/',
                                          f'similar_patients_scores_{note}_{date_time_str}.csv'), index=False)
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")


if __name__ == "__main__":

    setting_top_n = 50
    setting_test_mode = False
    setting_csv_mode = True
    setting_score_mode = True

    setting_start = 0
    setting_end = 1000

    setting_note = f"{setting_start}-{setting_end}"
    # setting_database_path = dir_now+'/../../../data/test/test_mimic_iv_30000.db'
    # setting_database_path = dir_now+'/../../../data/raw/mimic-iv-link/mimic4.db'
    setting_database_path = dir_now + '/../../data/raw/mimic-iv-link/all_csv' # need config

    # Testing case
    # setting_single_patient_id = [24117249]  # 1
    # setting_single_patient_id = [15460177]  # 1000
    # setting_single_patient_id = [19443249]  # 30000

    # Real loop

    # setting_target_patients_id_file_path = dir_now + '/../../data/find_similar_patients/5_3-40_prescriptions.json'
    # setting_source_patients_id_file_path = dir_now + '/../../data/find_similar_patients/1_live_hadm_id.json'

    setting_source_patients_id_file_path = dir_now + '/../../data/find_similar_patients/2_in_all_criteria.json'
    setting_target_patients_id_file_path = dir_now + '/../../data/find_similar_patients/5_3-40_prescriptions.json'

    # Open the file and read its content
    with open(setting_target_patients_id_file_path, 'r') as file:
        patient_ids = json.load(file)
        patient_ids = [np.int64(patient_id) for patient_id in patient_ids]
        patient_ids = patient_ids[setting_start:setting_end]

    # Random 200
    patient_ids = random.sample(patient_ids, 200)
    setting_single_patient_id = patient_ids

    # main(setting_test_mode, database_path=setting_database_path, top_n=setting_top_n)

    if setting_csv_mode:
        # Fetch target_hadm_ids by setting_single_patient_id

        target_hadm_ids = setting_single_patient_id

        # Fetch all_hadm_ids by all_patient_ids
        # Option1 only select patient with discharge summary
        with open(setting_source_patients_id_file_path, 'r') as file:
            all_hadm_ids = json.load(file)
            all_hadm_ids = [np.int64(patient_id) for patient_id in all_hadm_ids]

        # Option2 select everyone in EHR note
        # all_hadm_ids = ehr_similarity.fetch_all_hadm_id()

        ehr_similarity = EHRSimilarity(setting_database_path,
                                       top_n=setting_top_n,
                                       csv_mode=setting_csv_mode,
                                       split_bins=True,
                                       score_mode=setting_score_mode)

        # Test setting
        # ehr_similarity.find_top_similar_hadm_ids(hadm_ids=target_hadm_ids, single_hadm_id=target_hadm_ids)

        # Full setting
        ehr_similarity.find_top_similar_hadm_ids(hadm_ids=all_hadm_ids, single_hadm_id=target_hadm_ids)
        if ehr_similarity.split_bins:
            ehr_similarity.save_to_csv_bins(note=setting_note)
        elif ehr_similarity.with_score:
            ehr_similarity.save_to_csv_with_scores(note=setting_note)
        else:
            ehr_similarity.save_to_csv(note=setting_note)

    else:
        # DB mode for SQL
        # Author notes
        print("!!!!!!!!!!!!! Author note: SQL DB mode is very slow, which is the reason why we use csv to load. ")
        print("!!!!!!!!!!!!! Please make sure your RAM is enough to run")
        ehr_similarity = EHRSimilarity(setting_database_path, top_n=setting_top_n)
        all_patient_ids = ehr_similarity.fetch_all_subject_id()
        ehr_similarity.find_top_similar_hadm_ids(all_patient_ids, single_hadm_id=setting_single_patient_id)

    print("Done")
