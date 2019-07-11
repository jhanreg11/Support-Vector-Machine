import numpy as np
import pandas as pd
from svm import svm

df = pd.read_csv('data/pulsar_stars.csv')
npa = np.asarray(df)
for i in npa:
    if not i[8]:
        i[8] = -1

train_X = npa[:10000][:7]
train_Y = npa[:10000][8]

test_X = npa[10000:][:7]
test_Y = npa[10000:][8]

w = svm(train_X, train_Y, .00001, 10)

errors, tot = 0, 0
for x, y in zip(test_X, test_Y):
    if y * np.dot(x, w) < 1:
        errors += 1
    tot += 1

print("testing error percentage: ", 100 * (errors / tot))