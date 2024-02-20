import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def plot_vectors(pickle_file):
    # Load vectors from pickle file
    with open(pickle_file, 'rb') as f:
        vectors = pickle.load(f)

    # Separate names and vectors
    names = list(vectors.keys())
    vectors = list(vectors.values())

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Plot the 2D vectors using seaborn
    plot = sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1])

    # Add text labels for each point
    for i, name in enumerate(names):
        plot.text(vectors_2d[i, 0], vectors_2d[i, 1], name)

    plt.show()
# import pickle
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.manifold import TSNE
#
# def plot_vectors(pickle_file):
#     # Load vectors from pickle file
#     with open(pickle_file, 'rb') as f:
#         vectors = pickle.load(f)
#
#     # Separate names and vectors
#     names = list(vectors.keys())
#     vectors = list(vectors.values())
#
#     # Reduce dimensionality to 3D using t-SNE
#     tsne = TSNE(n_components=3)
#
#     # Convert list of lists to 2D NumPy array
#     vectors = np.array(vectors)
#
#     # Assuming that vectors is a list of vectors
#     n_samples = len(vectors)
#
#     # Set perplexity to a value less than n_samples
#     perplexity = n_samples - 1 if n_samples > 1 else 1
#
#     tsne = TSNE(n_components=3, perplexity=perplexity)
#     vectors_3d = tsne.fit_transform(vectors)    # Create a DataFrame for the 3D vectors
#     df = pd.DataFrame(vectors_3d, columns=['x', 'y', 'z'])
#     df['name'] = names
#
#     # Plot the 3D vectors using Plotly
#     fig = px.scatter_3d(df, x='x', y='y', z='z', hover_data=['name'])
#     fig.show()