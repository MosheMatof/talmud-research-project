from classification import generate_vectors
from utilles import generate_csv_from_txt

generate_csv_from_txt('./test.csv')
generate_vectors('./test.csv', 'content', 'vectors.pkl')
# plot_vectors('vectors.pkl')