import numpy as np
import pandas as pd

# TODO: Add feature to try out diff rate/epoch number and to stop training once training error hits 0
def svm(X, y, rate, epochs):
    """returns linear weights for classifier
    X - a list of feature vectors
    y - classification for each feature vector
    """

    w = np.zeros(len(X[0]))
    reg = 1 / epochs

    for e in range(epochs):
        errors = 0
        for i, x in enumerate(X):
            yi = y[i]
            if yi * np.dot(x, w) < 1:
                w += rate * (yi * x - (2 * reg * w))
                errors += 1
            else:
                w += rate * (-2 * reg * w)
        print("Epoch", e, "update: \nerror %", 100 * (errors / len(X)), " \nclassifier weights:", w, '\n')
    return w


