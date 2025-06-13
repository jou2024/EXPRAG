import os
import shutil

dir_now = os.path.dirname(os.path.abspath(__file__))

# Path to the folder containing text files
text_source_folder_path = dir_now + "/../../../" + "data/raw/embeddings/all_text/"
text_batch_folder_path = dir_now + "/../../../" + "data/raw/embeddings/all_batch/"



def batch_copy_txt_files(source_folder="all_text", target_folder="all_batch", batch_size=1000):
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List all .txt files
    all_txt_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.txt')])

    for i, file_name in enumerate(all_txt_files):
        batch_number = (i // batch_size) + 1  # Start from 1
        batch_folder = os.path.join(target_folder, str(batch_number))
        os.makedirs(batch_folder, exist_ok=True)

        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(batch_folder, file_name)

        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(all_txt_files)} files into batches of {batch_size} (folders named 1, 2, 3, ...).")

if __name__ == "__main__":
    batch_copy_txt_files(source_folder=text_source_folder_path, target_folder=text_batch_folder_path)
