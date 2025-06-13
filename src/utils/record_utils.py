import csv
import os
from datetime import datetime


class ResultProcessor:
    def __init__(self, name_prefix, directory="../data/result_csv", is_long_name=True):
        self.name_prefix = name_prefix
        self.directory = directory
        self.data = {}
        self.column_order = []
        self.is_long_name = is_long_name
        # Ensure the directory exists
        os.makedirs(self.directory, exist_ok=True)

    def add_data(self, dict_data, column_name):
        """Add dictionary data with a specified column name."""
        if column_name.endswith("_dict"):
            column_name = column_name[:-5]  # Removes the last 5 characters "_dict"
        if not self.column_order:
            # This is the first dictionary, set the order of the keys
            self.column_order = list(dict_data.keys())
        self.data[column_name] = dict_data

    def add_multiple_data(self, **kwargs):
        """Adds multiple sets of dictionary data with their associated column names."""
        for column_name, dict_data in kwargs.items():
            self.add_data(dict_data, column_name)

    def generate_csv(self):
        """Generate a CSV file with the current data."""
        filename = self._generate_filename()
        filepath = os.path.join(self.directory, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            headers = ['question_id'] + list(self.data.keys())
            writer.writerow(headers)
            # Write the data rows
            for key in self.column_order:
                row = [key] + [self.data[col].get(key, '') for col in self.data]
                writer.writerow(row)
        print(f"CSV file has been created: {filepath}")

    def _generate_filename(self):
        """Generate a filename based on the dataset name, current datetime, and column names."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_long_name:
            columns_str = "_".join(self.data.keys())
        else:
            columns_str =""
        filename = f"{self.name_prefix}_{date_str}_{columns_str}.csv"
        return filename


# Example usage:
if __name__ == "__main__":
    processor = ResultProcessor("test_dataset", "../../data/result_csv")
    chosen_answers = {0: 'A', 1: 'B', 2: 'C'}
    correct_answers = {0: 'A', 1: 'B', 2: 'C'}
    patient_id = {0: 15455707, 1: 11801858, 2: 15648022}
    question = {
        0: "The patient was fairly stabilized, with pain under control, consuming a regular diet, and able to walk and relieve himself without assistance. This indicates that the patient's condition was good at the time of discharge in terms of vital signs, pain management, and mobility.",
        1: 'The patient was treated with cardiac catheterization and had a BMS successfully deployed leading to restored blood flow. This is the most common and effective treatment for STEMI/CAD, as it helps to open up blocked arteries and improve blood flow to the heart muscle.',
        2: 'The patient underwent pipeline embolization for their right paraclinoid ICA aneurysm, as mentioned in the intervention performed. Additionally, the patient was discharged in a clear and coherent state, indicating a successful procedure and recovery.'
    }

    processor.add_multiple_data(
        Patient_id=patient_id, Chosen=chosen_answers, Correct=correct_answers, Question=question
    )
    processor.generate_csv()
