import os
import re

DISCHARGE_NOTE_SECTIONS = {
  "Patient Information": [
    # "Name",
    # "Admission Date",
    # "Discharge Date",
    # "Date of Birth",
    # "Sex",
    # "Service",
    # "Attending",
    "Allergies"
  ],
  "Presenting Condition": [
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History"
  ],
  "Clinical Assessment": [
    "Physical Exam",
    # "Pertinent Results"  #TODO
  ],
  "Treatment Plan": [
    "Major Surgical or Invasive Procedure"
  ],
  "In-Hospital Progress": [
    "Pertinent Results", # TODO Split
    "Brief Hospital Course",
    "Medications on Admission"
  ],
  "Discharge Summary": [
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition"
  ],
  "Post-Discharge Instructions": [
    "Discharge Instructions",
    "Followup Instructions"
  ]
}



class DischargeSummarySplitter:
    def __init__(self, text, only_instruction=False, only_medications=False, only_diagnosis=False):
        self.text = text
        self.only_instruction = only_instruction
        self.only_medications = only_medications
        self.only_diagnosis = only_diagnosis

        if only_medications:
            self.parts = {"Discharge Medications": []}
        elif only_diagnosis:
            self.parts = {"Discharge Diagnosis": []}
        elif only_instruction:
            self.parts = {"Discharge Summary": []}
        else:
            self.parts = {
                key: "" if isinstance(value, list) else value
                for key, value in DISCHARGE_NOTE_SECTIONS.items()
            }

    def remove_unuseful_parts(self):
        # Remove name, unit number, admission/discharge date, etc.
        patterns = [r"Name:.*", r"Unit No:.*", r"Admission Date:.*", r"Discharge Date:.*",
                    r"Date of Birth:.*", r"Sex:.*", r"Attending:.*", r"Service:.*"]
        for pattern in patterns:
            self.text = re.sub(pattern, "", self.text)

    def split_into_parts(self):
        # Define shared keywords for each part
        # Split the document into separate notes using the ',), (' delimiter
        sections = re.split(r"',\), \('", self.text)

        for section in sections:
        
            if self.only_instruction:
                # For Discharge Instructions and Followup Instructions
                discharge_match = re.search(r"Discharge Instructions:.*?(?=Followup Instructions:)", section, re.DOTALL)
                followup_match = re.search(r"Followup Instructions:.*?(?=\',\))", section, re.DOTALL)

                if discharge_match:
                    discharge_content = "Discharge Instructions:\n" + discharge_match.group(0).strip() + "\n"
                    self.parts["Discharge Summary"].append(discharge_content)
                if followup_match:
                    followup_content = followup_match.group(0).strip()
                    # Filter out if Followup Instructions are too short (e.g., if it's just underscores)
                    length_followup_instruction = len(followup_content.replace("_", "").strip())
                    if length_followup_instruction > 30:  # Adjust length threshold as needed
                        self.parts["Discharge Summary"].append(followup_content)
                    else:
                        print(" No Followup Instruction")
            elif self.only_medications:
                # For Discharge medications
                medications_match = re.search(r"Discharge Medications:.*?(?=Discharge Disposition:)", section, re.DOTALL)

                if medications_match:
                    medications_content = "Discharge Medications:\n" + medications_match.group(0).strip() + "\n"
                    self.parts["Discharge Medications"].append(medications_content)
            elif self.only_diagnosis:
                # For Discharge medications
                diagnosis_match = re.search(r"Discharge Diagnosis:.*?(?=Discharge Condition:)", section, re.DOTALL)

                if diagnosis_match:
                    diagnosis_content = "Discharge Diagnosis:\n" + diagnosis_match.group(0).strip() + "\n"
                    self.parts["Discharge Diagnosis"].append(diagnosis_content)
            else:
                keywords = DISCHARGE_NOTE_SECTIONS
                all_keywords_pattern = "|".join(
                    [re.escape(keyword) for section in keywords.values() for keyword in section])
                # all_keywords_pattern = all_keywords_pattern.join("|".join("Medications on Admission")).join("|".join("Pertinent Results"))
                for part, key_list in keywords.items():
                    for keyword in key_list:
                        # Updated regex pattern to match until the next keyword in DISCHARGE_NOTE_SECTIONS
                        pattern = rf"{keyword}:.*?(?=\n({all_keywords_pattern}):|\Z)"
                        match = re.search(pattern, self.text, re.DOTALL)
                        if match:
                            # Add matched content to the current part
                            self.parts[part] += match.group(0).strip() + "\n"
                            # Remove the matched text from self.text to avoid duplicate processing
                            self.text = self.text.replace(match.group(0), "")

    def handle_missing_parts(self):
        # Handle cases where some parts might be missing
        for part, content in self.parts.items():
            if not content:
                self.parts[part] = f"{part}: Not Found\n"

    def save_to_files(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for part, content in self.parts.items():
            filename = os.path.join(output_dir, f"{part.replace(' ', '_').lower()}.txt")
            with open(filename, 'w') as file:
                file.write(content)

    def process_summary(self, output_dir=None):
        self.remove_unuseful_parts()
        self.split_into_parts()
        self.handle_missing_parts()
        if output_dir:
            self.save_to_files(output_dir)
        return self.parts


def main():
    dir_now = os.path.dirname(os.path.abspath(__file__))
    input_file_dir = dir_now + '/../../../tmp/discharge_summary/'
    # Example usage
    # file_path = input_file_dir + "/retrieved_20979675.txt"  # only pre diagnosis in Pertinent Results
    file_path = input_file_dir + "/retrieved_25690125.txt"  # both pre diagnosis and post diagnosis data in Pertinent Results

    output_dir = input_file_dir + "/split_summary/"

    with open(file_path, 'r') as file:
        text = file.read()

    only_instruction_splitter = DischargeSummarySplitter(text, only_instruction=True)
    parts = only_instruction_splitter.process_summary()

    overall_splitter = DischargeSummarySplitter(text)
    overall_parts = overall_splitter.process_summary()

    # Print parts to console for testing
    for part, content in overall_parts.items():
        print(f"---- {part} ----")
        print(content)
        print()


if __name__ == "__main__":
    main()
