import csv
import os
dir_now = os.path.dirname(os.path.abspath(__file__))


def save_first_row_as_files(file_path, output_dir):
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the CSV file
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Get the first row
            first_row = next(reader, None)
            
            if first_row:
                for column_name, value in first_row.items():
                    # Create a file for each column
                    sanitized_column_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in column_name)
                    output_file = os.path.join(output_dir, f"{sanitized_column_name}.txt")
                    
                    with open(output_file, mode='w', encoding='utf-8') as output:
                        output.write(value)
                        print(f"Saved: {output_file}")
            else:
                print("The CSV file is empty or does not have rows.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_rows_with_hadm_id_as_files(file_path, output_dir, max_rows="all"):
    """
    Save rows from a CSV file to files. Each file is named after the value in the 'hadm_id' column,
    and its content is the value from the 'text' column.

    Parameters:
        file_path (str): Path to the CSV file.
        output_dir (str): Directory to save the files.
        max_rows (int or str): Number of rows to process. Use "all" to process all rows.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_number, row in enumerate(reader, start=1):
                if max_rows != "all" and row_number > int(max_rows):
                    break

                # Get 'hadm_id' and 'text' columns
                hadm_id = row.get("hadm_id", "").strip()
                text = row.get("text", "").strip()
                
                if hadm_id and text:  # Ensure both columns are present
                    output_file = os.path.join(output_dir, f"{hadm_id}.txt")
                    with open(output_file, mode='w', encoding='utf-8') as output:
                        output.write(text)
                        print(f"Saved: {output_file}")
                else:
                    print(f"Row {row_number} skipped: Missing 'hadm_id' or 'text'")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Prompt user for input file
    file_path = dir_now + "/../../../" + "data/raw/mimic-iv-link/all_csv/discharge.csv"
    dir_text = dir_now + "/../../../" + "data/raw/embeddings/all_text/"
    save_rows_with_hadm_id_as_files(file_path, dir_text)
