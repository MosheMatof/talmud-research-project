from classification import generate_vectors, train_multiple_classifiers
from utilles import generate_csv_from_txt
from visualizer import plot_vectors
import pandas as pd

generate_csv_from_txt('./sfarim.csv', 'sfarim/suka')
df = pd.read_csv('./sfarim.csv')
generate_vectors(df,'name', 'content', 'vectors.pkl')
#train_multiple_classifiers('vectors.pkl')
plot_vectors('vectors.pkl')
