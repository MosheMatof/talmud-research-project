import PySimpleGUI as sg
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from visualizer import plot_vectors
from classification import generate_vectors, train_multiple_classifiers
from utilles import generate_csv_from_txt

# Define your seaborn plotting function
def create_seaborn_plot(vectors):
    return plot_vectors(vectors, show_plot=False)

# Define layout
layout = [
    [sg.Text("Select file or folder:"), sg.In(key="filepath"), sg.FileBrowse(key='file_browse', file_types=(("Text Files", "*.txt"),)), sg.FolderBrowse(key='folder_browse', disabled=True)],
    [sg.Radio("Single Text File", "upload_type", key="single_file", default=True, enable_events=True), sg.Radio("Folder of Folders", "upload_type", key="folder", enable_events=True)],
    [sg.Button("Upload")],
    [sg.Canvas(key="plot")],
]

# Create the window
window = sg.Window("File Upload and Plot", layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    
    if event == "single_file":
        window['file_browse'].update(disabled=False)
        window['folder_browse'].update(disabled=True)
    elif event == "folder":
        window['file_browse'].update(disabled=True)
        window['folder_browse'].update(disabled=False)

    if event == "Upload":
        filepath = values["filepath"]
        if  values["single_file"]:
            # Read data from single text file
            with open(filepath, "r") as f:
                data = [float(line.strip()) for line in f]

            # Create seaborn plot and convert to figure
            figure = create_seaborn_plot(data)

        elif values["folder"]:
            # Process folder of folders (same logic as before)
            generate_csv_from_txt('sfarim.csv', folder_path=filepath)
            df = pd.read_csv('sfarim.csv')
            generate_vectors(df, 'name', 'content', 'vectors.pkl')

            all_data = 'vectors.pkl'

            # Combine data and create seaborn plot
            if all_data:
                figure = create_seaborn_plot(all_data)
            else:
                sg.popup("No text files found in the selected folder.")

        # Convert figure to a PySimpleGUI image
        img_bytes = sg.FigureCanvasTkAgg(figure).get_tk_image().export_bytes()
        image = sg.Image(data=img_bytes)

        # Update the canvas with the image
        window["plot"].update(image)

window.close()
