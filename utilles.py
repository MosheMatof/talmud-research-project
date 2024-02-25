import os
import pandas as pd

files_path = 'sfarim/Baba_kama/baycha'
def generate_csv_from_txt(csv_file_path, folder_path = files_path):
    data = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                # Get the directory name and append it to the filename
                directory_name = os.path.basename(dirpath)
                new_filename = filename + "_" + directory_name
                data.append((new_filename, content))
    df = pd.DataFrame(data, columns=['name', 'content'])
    df.to_csv(csv_file_path, index=False)
