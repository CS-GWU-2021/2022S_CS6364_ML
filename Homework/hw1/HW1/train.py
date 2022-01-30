from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def find_k(X, y):
    k_range = range(1,20)
    k_error = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_error.append(1-scores.mean())

    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()
