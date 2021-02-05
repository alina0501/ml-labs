import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from pandas import DataFrame
from sklearn import datasets, metrics
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import seaborn as sns

moons_X, moons_y = datasets.make_moons(n_samples=1200, noise=0.05)

digits_X, digits_y = datasets.load_digits(return_X_y=True)

sns.set()

digits_num_clusters = 10
moons_num_clusters = 2

# fig = plt.figure()
# fig.subplots_adjust()
#
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits_X[i].reshape((8,8)), cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
#
pca = PCA(n_components=2)
digits_projection = pca.fit_transform(digits_X)
# plt.scatter(digits_projection[:, 0], digits_projection[:, 1])
# plt.show()
#
# plt.scatter(moons_X[:, 0], moons_X[:, 1])
# plt.show()

digits_spectral = SpectralClustering(n_clusters=digits_num_clusters,
                                     eigen_solver='arpack',
                                     affinity="nearest_neighbors")

moons_spectral = SpectralClustering(n_clusters=moons_num_clusters,
                                    eigen_solver='arpack',
                                    affinity="nearest_neighbors")

digits_clusters = digits_spectral.fit_predict(digits_X)
moons_clusters = moons_spectral.fit_predict(moons_X)


# fig = plt.figure()
# for c in range(digits_num_clusters):
#     ax = fig.add_subplot(4, 3, 1 + c, xticks=[], yticks=[])
#     f = digits_X[digits_clusters == c, :]
#     cluster_center = np.mean(f, axis=0)
#     ax.imshow(cluster_center.reshape((8, 8)), cmap=plt.cm.binary)
# plt.show()

# skplt.decomposition.plot_pca_2d_projection(pca, digits_X, digits_clusters, title='Digits Clusters 2-D Projection')
# plt.show()
#
# df = DataFrame(dict(x=moons_X[:, 0], y=moons_X[:, 1], label=moons_clusters))
# colors = {0: 'orange', 1: 'purple'}
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# plt.show()



def clustering_metrics(moons_labels, digits_labels):
    data = {'moons': [metrics.adjusted_rand_score(moons_y, moons_labels),
                      metrics.davies_bouldin_score(moons_X, moons_labels),
                      metrics.silhouette_score(moons_X, moons_labels),
                      moons_num_clusters],
            'digits': [metrics.adjusted_rand_score(digits_y, digits_labels),
                       metrics.silhouette_score(digits_X, digits_labels),
                       metrics.davies_bouldin_score(digits_X, digits_labels),
                       digits_num_clusters]}

    spectral_metrics = DataFrame(data=data, index=['Adjusted Rand Index', 'Silhouette coefficient',
                                                   'Davies-Bouldin index', 'Estimated number of clusters'])
    return spectral_metrics

i = 0
digits_spectrals = []
moons_spectrals = []

print('--------------Tunning How to Construct the Affinity Matrix-------------------')
for moons_affin, digits_affin in [['rbf', 'nearest_neighbors'],
                                  ['laplacian', 'cosine'],
                                  ['poly', 'poly'],
                                  ['nearest_neighbors', 'linear']]:
    digits_spectrals.append(SpectralClustering(n_clusters=digits_num_clusters,
                                         eigen_solver='arpack',
                                         affinity=digits_affin).fit_predict(digits_X))
    moons_spectrals.append(SpectralClustering(n_clusters=moons_num_clusters,
                                        eigen_solver='arpack',
                                        affinity=moons_affin).fit_predict(moons_X))
    print(f"*****************Metrics of Spectral Clustering****************************\n"
          f"Digits model #{i+1}: \n {digits_num_clusters} clusters, {digits_affin} affinity"
          f"\nMoons model #{i+1}: \n {moons_num_clusters} clusters, {moons_affin} affinity"
          f"\n{clustering_metrics(moons_spectrals[i], digits_spectrals[i])}")
    df = DataFrame(dict(x=moons_X[:, 0], y=moons_X[:, 1], label=moons_spectrals[i]))
    colors = {0: 'orange', 1: 'purple'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key],
                   title=f'Spectral Clustering on Moons, affinity = {moons_affin}')
    plt.show()
    skplt.metrics.plot_silhouette(digits_X, digits_spectrals[i],
                                  title=f'Spectral Clustering on Digits Silhouette Analysis, affinity = {digits_affin}')
    plt.show()
    i += 1

digits_spectrals.clear()
moons_spectrals.clear()

digits_spectrals = []
moons_spectrals = []
i = 0

# print('\n\n--------------Tunning Number of Clusters-------------------\n\n')
# for moons_num, digits_num in zip([2, 3, 4, 5, 6], [7, 8, 9, 10, 11]):
#     digits_spectrals.append(SpectralClustering(n_clusters=digits_num,
#                                          eigen_solver='arpack',
#                                          affinity='nearest_neighbors').fit_predict(digits_X))
#     moons_spectrals.append(SpectralClustering(n_clusters=moons_num,
#                                         eigen_solver='arpack',
#                                         affinity='nearest_neighbors').fit_predict(moons_X))
#     print(f"*****************Metrics of Spectral Clustering****************************\n"
#           f"Digits model #{i+1}: \n {digits_num} clusters, nearest_neighbors affinity"
#           f"\nMoons model #{i+1}: \n {moons_num} clusters, nearest_neighbors affinity"
#           f"\n{clustering_metrics(moons_spectrals[i], digits_spectrals[i])}")
#     df = DataFrame(dict(x=moons_X[:, 0], y=moons_X[:, 1], label=moons_spectrals[i]))
#     colors = {0: 'orange', 1: 'purple', 2: 'cyan', 3: 'yellow', 4: 'brown', 5: 'blue'}
#     fig, ax = plt.subplots()
#     grouped = df.groupby('label')
#     for key, group in grouped:
#         group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key],
#                    title=f'Spectral Clustering on Moons, affinity = nearest_neighbors, {moons_num} clusters')
#     plt.show()
#     skplt.metrics.plot_silhouette(digits_X, digits_spectrals[i],
#                                   title=f'Spectral Clustering on Digits Silhouette Analysis , '
#                                         f'{digits_num} clusters')
#     skplt.metrics.plot_silhouette(moons_X, moons_spectrals[i],
#                                   title=f'Spectral Clustering on Moons Silhouette Analysis , '
#                                         f'{moons_num} clusters')
#     i += 1
# plt.show()
# digits_spectrals.clear()
# moons_spectrals.clear()

print(f'Shapes before deletion of elements:\n'
      f'   moons_X: {moons_X.shape}\n'
      f'   moons_y: {moons_y.shape}\n'
      f'   digits_X: {digits_X.shape}\n'
      f'   digits_y: {digits_y.shape}\n')
print(f'Metrics before deletion:\n{clustering_metrics(moons_clusters, digits_clusters)}')
plt.scatter(moons_X[:, 0], moons_X[:, 1], c=moons_clusters, cmap='PuOr')
plt.show()
skplt.decomposition.plot_pca_2d_projection(pca, digits_X, digits_clusters,
                                           title='Digits Clusters 2-D Projection Before Deletion')
plt.show()

fig = plt.figure()
fig.suptitle('Cluster Centres Before Deletion')
for c in range(digits_num_clusters):
    ax = fig.add_subplot(4, 3, 1 + c, xticks=[], yticks=[])
    f = digits_X[digits_clusters == c, :]
    cluster_center = np.mean(f, axis=0)
    ax.imshow(cluster_center.reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

count = 0
for count in range(500):
    deletion_position_moons = np.random.randint(0, len(moons_X))
    deletion_position_digits = np.random.randint(0, len(digits_X))
    digits_X = np.delete(digits_X, deletion_position_digits, axis=0)
    moons_X = np.delete(moons_X, deletion_position_moons, axis=0)
    digits_y = np.delete(digits_y, deletion_position_digits, axis=0)
    moons_y = np.delete(moons_y, deletion_position_moons, axis=0)

print(f'Shapes after deletion of elements:\n'
      f'   moons_X: {moons_X.shape}\n'
      f'   moons_y: {moons_y.shape}\n'
      f'   digits_X: {digits_X.shape}\n'
      f'   digits_y: {digits_y.shape}\n')

digits_clusters = digits_spectral.fit_predict(digits_X)
moons_clusters = moons_spectral.fit_predict(moons_X)

plt.scatter(moons_X[:, 0], moons_X[:, 1], c=moons_clusters, cmap='PuOr')
plt.show()
skplt.decomposition.plot_pca_2d_projection(pca, digits_X, digits_clusters, title='Digits Clusters 2-D Projection After Deletion')
plt.show()
print(f'Metrics after deletion:\n{clustering_metrics(moons_clusters, digits_clusters)}')

fig = plt.figure()
fig.suptitle('Cluster Centres After Deletion')
for c in range(digits_num_clusters):
    ax = fig.add_subplot(4, 3, 1 + c, xticks=[], yticks=[])
    f = digits_X[digits_clusters == c, :]
    cluster_center = np.mean(f, axis=0)
    ax.imshow(cluster_center.reshape((8, 8)), cmap=plt.cm.binary)
plt.show()



