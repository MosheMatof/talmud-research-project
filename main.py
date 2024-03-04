from classification import generate_vectors, train_multiple_classifiers
from utilles import generate_csv_from_txt
from visualizer import plot_vectors
import pandas as pd
import os

# Get a list of all subdirectories in 'sfarim'
subdirectories = [d for d in os.listdir('sfarim') if os.path.isdir(os.path.join('sfarim', d))]

# Iterate over each subdirectory
for subdirectory in subdirectories:
    # Generate CSV and vectors for each subdirectory
    generate_csv_from_txt('./sfarim.csv', os.path.join('sfarim', subdirectory))
    df = pd.read_csv('./sfarim.csv')
    vectors_file = os.path.join('vectors',subdirectory + '.pkl')
    generate_vectors(df, 'name', 'content', vectors_file)
    #train_multiple_classifiers(vectors_file)
    plot_vectors(vectors_file)
