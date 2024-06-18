import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering

# Function to compute the Fréchet distance between two sequences
def frechet_distance(X, Y):
    n = len(X)
    m = len(Y)
    ca = np.ones((n, m)) * -1

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(X[0] - Y[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), np.linalg.norm(X[i] - Y[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), np.linalg.norm(X[0] - Y[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)), np.linalg.norm(X[i] - Y[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return _c(n-1, m-1)

# Load the dataset
file_path = 'E:\DTW_Frechet_distance\data\jan_feb_2024_subset.csv'
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

# Save the results
df.to_csv('E:\\sigai\\clustered_data.csv')
