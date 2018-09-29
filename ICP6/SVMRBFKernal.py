# use the SVM with RBF kernel on the same dataset.
from sklearn.svm import SVC
from sklearn import datasets
# Loading the dataset
irisdataset = datasets.load_iris()
# getting the data and response of the dataset
x = irisdataset.data
y = irisdataset.target
# Creating the Model
model = SVC(kernel='rbf', C=1, gamma=1)
model.fit(x, y)
print(model.predict([[7, 2, 3, 4], [2, 3, 4, 5]]))
#Predict by passing random values
print(model.score(x, y))