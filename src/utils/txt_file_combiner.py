import os
import re

class TxtFileCombiner:
    """
    This class takes a directory as input, combines all text files inside it,
    and saves the combined content into a single text file. Each section in the
    output file is separated by the corresponding file name.
    """

    def __init__(self, input_dir: str, output_file: str):
        """
        Initialize the class with input directory and output file path.

        Args:
            input_dir (str): Path to the directory containing the .txt files.
            output_file (str): Path where the combined output file will be saved.
        """
        self.input_dir = input_dir
        self.output_file = output_file

    def combine_txt_files(self, similar_id_list) -> str:
        """
        Combines the contents of all .txt files in the input directory into a single file.
        Each file's content is separated by its file name as a header.

        Returns:
            str: The path to the combined output file.
        """
        try:
            with open(self.output_file, 'w') as outfile:
                count = 0
                for filename in sorted(os.listdir(self.input_dir)):
                    match = re.search(r"_(\d+)\.txt$", filename)
                    # Process only .txt files TODO: embedding scan uses list of int; while map file uses list of str
                    if match and filename.endswith('.txt') and filename.startswith('retrieved') and ((int(match.group(1)) in similar_id_list) or (match.group(1) in similar_id_list)):
                        count += 1
                        file_path = os.path.join(self.input_dir, filename)
                        # Write the file name as a section header
                        outfile.write(f"===== {filename} =====\n")
                        with open(file_path, 'r') as infile:
                            content = infile.read()
                            outfile.write(content + "\n\n")
                saved_file_path = "/".join(self.output_file.split(os.sep)[-3:])
                print(f"Combined {count} text saved to: {saved_file_path}")
            return self.output_file
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

dir_now = os.path.dirname(os.path.abspath(__file__))
project_root_dir = dir_now + "/../../"

# Example Usage:
if __name__ == "__main__":
    # Initialize the TxtFileCombiner with input directory and output file path
    input_directory = project_root_dir + "/tmp/discharge_summary/similar_patients/20531549similar/"
    output_file_path = project_root_dir + "/data/test/combined_20531549similar.txt"

    # Create an instance of the combiner and run it
    combiner = TxtFileCombiner(input_directory, output_file_path)
    combined_file_path = combiner.combine_txt_files()

    # Output the path of the combined file
    print(f"Combined file path: {combined_file_path}")
