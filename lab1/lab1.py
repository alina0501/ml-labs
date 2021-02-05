import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dist(a, b):
    return np.around(np.linalg.norm(a-b, axis=1), decimals=4)


def calc(a, b, c):
    Outp = np.empty((G, M), dtype=float)
    multipl = np.around(np.dot(a, b), decimals=4)
    temp = np.around(np.sum(a, axis=1), decimals=4)
    for g in range(c):
        Outp[:, g] = np.around(multipl[:, g] / temp, decimals=4)
    return Outp


dataset = pd.read_csv('iris.csv')
dataset.describe()
T = np.array(dataset.iloc[:, [1, 2]].values)
N = T.shape[0]  # number of training examples
M = T.shape[1]  # number of features. Here M=2
n_iter = 5
G = 5  # number of clusters
U = np.zeros((G, N), dtype=float)  # GxN

for i in range(N):
    rand = np.random.randint(0, G)
    U[rand, i] = 1.0

C = calc(U, T, M)
plt.scatter(T[:, 0], T[:, 1], c='black', label='data')
plt.scatter(C[:, 0], C[:, 1], s=300, c='yellow', label='Centroids')
plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.legend()
plt.show()

for it in range(n_iter):
    EuclidianDistance = np.array([]).reshape(N, 0)
    for k in range(G):
        tempDist = dist(T, C[k, :])
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    arg = np.min(EuclidianDistance, axis=1)

    for j in range(N):
        U[EuclidianDistance[j, :] != arg[j], j] = 0.0
        U[EuclidianDistance[j, :] == arg[j], j] = 1.0

    C = calc(U, T, M)
    plt.scatter(T[:, 0], T[:, 1], c='black', label='data')
    plt.scatter(C[:, 0], C[:, 1], s=300, c='yellow', label='Centroids')
    plt.xlabel('sepal width')
    plt.ylabel('petal length')
    plt.legend()
    plt.show()

print(C)
