import numpy as np
import pandas as pd

def get_input():
    file_path = input("----------------------\nEnter the file path for the csv: ")
    start_index = input("----------------------\nEnter the starting column index of the feature columns (0 starting index): ")
    end_index = input("----------------------\nEnter ending column index for feature columns(0 starting index): ")
    class_index = input("----------------------\nEnter the index of the classification column(0 starting index): ")
    train_start_index = input("----------------------\nEnter the starting row index of the training data: ")
    train_end_index = input("----------------------\nEnter the ending index of the training data: ")
    test_start_index = input("----------------------\nEnter the starting row index of the testing data: ")
    test_end_index = input("----------------------\nEnter the ending index of the testing data: ")
    try:
        df = pd.read_csv(file_path)
        npa = np.asarray(df)
        train_X = npa[int(train_start_index):int(train_end_index)][int(start_index):int(end_index)+1]
        train_y = npa[int(train_start_index):int(train_end_index)][int(class_index)]
        test_X = npa[int(test_start_index):int(test_end_index)][int(start_index):int(end_index)+1]
        test_y = npa[int(test_start_index):int(test_end_index)][int(class_index)]

    except:
        print("\n\nOops! something was wrong with your input. Please try again...")
        return get_input()

    rate = input('Enter learning rate (enter \"u\" if you don\'t know) : ')
    epochs = input('Enter number of iterations you would like the model to train (\"u\" if unknown): ')

    return [train_X, train_y, test_X, test_y, float(rate), int(epochs)]

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
        if e % (epochs//10) == 0:
            print("\nEpoch ", e, "\nupdate: error %", 100 * (errors / len(X)), " \nclassifier weights: ", w)
    return w

data_segments = get_input()

w = svm(data_segments[0], data_segments[1], data_segments[4], data_segments[5])

errors, tot = 0, 0
for x, y in zip(data_segments[2], data_segments[3]):
    if y * np.dot(x, w) < 1:
        errors += 1
    tot += 1

print("testing error percentage: %", 100 * (errors / tot),)

