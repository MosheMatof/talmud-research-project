from classification import generate_vectors
from utilles import generate_csv_from_txt
from visualizer import plot_vectors
import pandas as pd

generate_csv_from_txt('./test.csv')
df = pd.read_csv('./test.csv')
generate_vectors(df,'name', 'content', 'vectors.pkl')
plot_vectors('vectors.pkl')