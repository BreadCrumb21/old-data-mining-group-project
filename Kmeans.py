import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.preprocessing import OrdinalEncoder


col_names = ['type','education','status','service','relation','race','gender','location','capital']
# load dataset
test_data = pd.read_csv("adultTestCorrected.test", header=None, names=col_names)
# print(test_data.head())

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[["type","education","status","service","relation","race","gender","location","capital"]])
test_data[["type","education","status","service","relation","race","gender","location","capital"]] = enc.fit_transform(
    test_data[["type","education","status","service","relation","race","gender","location","capital"]])


# Define the values of k
k_values = [3, 5, 10]

# Loop through different values of k
for k in k_values:
    # Create a KMeans instance with the specified number of clusters (k)
    kmeans = KMeans(n_clusters=k, random_state=0)
    
    # Fit the model to the training data
    kmeans.fit(test_data)
    
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    
    # Print the centroids for the current value of k
    print(f"Centroids for k={k}:")
    print(centroids)
