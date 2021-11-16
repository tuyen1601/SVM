# imports lib
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# configuration options
random_seed = 42
n_samples = 60  # total number of points
n_features = 2  # the number of features
centers = [(0, 0), (3, 3)]  # center points location

# generation dataset
X, y = make_blobs(n_features=n_features, n_samples=n_samples, centers=centers)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed)

# save file and load dataset
np.save('./data.npy', (X_train, X_test, y_train, y_test))
X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)

# generate scatter plot for training data
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.title('dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
