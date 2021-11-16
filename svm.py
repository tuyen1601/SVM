from dataset import *

from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions

# init SVM classifier
clf = SVC(kernel='linear')

# fit dataset
clf = clf.fit(X_train, y_train)

# predict test
y_pred = clf.predict(X_test)

# generate confution matrix
matrix = plot_confusion_matrix(
    clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.title('confution matrix for classifier')
plt.show(matrix)
plt.show()

# get support vectors
support_vectors = clf.support_vectors_

# visualize support vectors
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.scatter(support_vectors[:, 0], support_vectors[:, 1])
plt.title('linear data with svc')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.show()
