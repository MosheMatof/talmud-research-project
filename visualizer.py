import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def plot_vectors(pickle_file):
    # Load vectors from pickle file
    with open(pickle_file, 'rb') as f:
        vectors = pickle.load(f)

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Plot the 2D vectors using seaborn
    sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1])
    plt.show()

# Usage
plot_vectors('vectors.pkl')