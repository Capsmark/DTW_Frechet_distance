import os

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering


def frechet_distance(x, y):
    n, m = len(x), len(y)
    ca = np.ones((n, m)) * -1

    # c for cols and r for rows
    def _c(c, r):
        if ca[c, r] > -1:
            return ca[c, r]
        if c == 0 and r == 0:
            ca[c, r] = np.linalg.norm(x[0] - y[0])
        elif c > 0 and r == 0:
            ca[c, r] = max(_c(c - 1, 0), np.linalg.norm(x[c] - y[0]))
        elif c == 0 and r > 0:
            ca[c, r] = max(_c(0, r - 1), np.linalg.norm(x[0] - y[r]))
        else:  # i > 0 and j > 0
            ca[c, r] = max(min(_c(c - 1, r), _c(c - 1, r - 1), _c(c, r - 1)), np.linalg.norm(x[c] - y[r]))
        return ca[c, r]

    return _c(n-1, m-1)


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

# Compute pairwise Fréchet distances
num_sequences = len(sequence_list)
distance_matrix = np.zeros((num_sequences, num_sequences))

for i in range(num_sequences):
    for j in range(i + 1, num_sequences):
        distance = frechet_distance(sequence_list[i], sequence_list[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance


# Convert the distance matrix to an affinity matrix using the RBF kernel
sigma = np.median(distance_matrix)  # or choose another suitable value for sigma
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