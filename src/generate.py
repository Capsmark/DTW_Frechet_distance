import os

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from joblib import Parallel, delayed


def frechet_distance(x, y):
    n, m = len(x), len(y)
    ca = np.ones((n, m)) * -1
    ca[0, 0] = np.linalg.norm(x[0] - y[0])

    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(x[i] - y[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(x[0] - y[j]))

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                np.linalg.norm(x[i] - y[j])
            )

    return ca[n - 1, m - 1]


# Load the dataset
repo_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(repo_folder, 'data', 'jan_feb_2024_subset.csv')
df = pd.read_csv(file_path)

# Convert datetime column to pandas datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.sort_index(ascending=True, inplace=True)

# Extract sequences based on 'index'
sequences = defaultdict(list)
for _, row in df.iterrows():
    sequences[row['index']].append(row['sigai'])

# Convert sequences dictionary to a list of sequences
sequence_list = list(sequences.values())


# Function to compute distances in parallel
def compute_distance(c, r):
    dist = frechet_distance(np.array(sequence_list[c]), np.array(sequence_list[r]))
    return c, r, dist


# Compute pairwise Fréchet distances in parallel
num_sequences = len(sequence_list)
distance_matrix = np.zeros((num_sequences, num_sequences))

results = Parallel(n_jobs=-1)(delayed(compute_distance)(col, row) for col in range(num_sequences) for row in range(col + 1, num_sequences))

for col, row, distance in results:
    distance_matrix[col, row] = distance
    distance_matrix[row, col] = distance

# Convert the distance matrix to an affinity matrix using the RBF kernel
sigma = np.median(distance_matrix[np.triu_indices(num_sequences, 1)])  # Exclude zeros from the diagonal
affinity_matrix = np.exp(-distance_matrix ** 2 / (2. * sigma ** 2))

# Apply Spectral Clustering
num_clusters = 5  # Set the number of clusters you want
spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
labels = spectral.fit_predict(affinity_matrix)

# Add cluster labels to the dataframe
df['cluster'] = df['index'].map(dict(zip(sequences.keys(), labels)))

# Ensure the export directory exists
export_dir = os.path.join(repo_folder, 'export')
os.makedirs(export_dir, exist_ok=True)

# Save the results
output_path = os.path.join(export_dir, 'clustered_data.csv')
df.to_csv(output_path)
