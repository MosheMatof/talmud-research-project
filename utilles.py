import os
import pandas as pd

files_path = './sfarim'
def generate_csv_from_txt(csv_file_path, folder_path = files_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
            data.append((filename, content))
    df = pd.DataFrame(data, columns=['name', 'content'])
    df.to_csv(csv_file_path, index=False)
