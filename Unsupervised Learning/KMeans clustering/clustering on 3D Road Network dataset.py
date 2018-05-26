from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE

df = pd.read_csv('3D_spatial_network.txt')
X = df.values

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(df)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters=3)

# kmeans.fit(df)
#
# labels = kmeans.predict(df)

scaler = Normalizer()

pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(df)

labels = pipeline.predict(df)

plt.scatter(labels,df.iloc[:,0],c=labels,alpha=0.5,s=1)
plt.show()

mergings = linkage(df,method='complete')
dendrogram(mergings,
           labels=labels,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(df)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,c=labels)
plt.show()




#crosstabulation won't work because this is an unlabeled dataset
# df_labels = pd.DataFrame({'labels':labels,'category':labels})
# print(pd.crosstab(df_labels['labels'],df_labels['category']))


# Assign the columns of new_points: xs and ys
xs = df.iloc[:,0]
ys = df.iloc[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5,s=5)

# Assign the cluster centers: centroids
centroids = kmeans.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
#
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=25)
plt.show()

plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s =10, c = 'red', label = 'category 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s =10, c = 'blue', label = 'category 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s =10, c = 'cyan', label = 'category 3')
plt.scatter(centroids_x,centroids_y, s = 30, color = 'yellow', label = 'centroid')
plt.title('clusters of features')
plt.legend()
plt.show()