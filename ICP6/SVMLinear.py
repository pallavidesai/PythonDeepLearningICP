# Implement linear SVM method using scikit library
from sklearn.svm import LinearSVC
from sklearn import datasets
# Loading the dataset
irisdataset = datasets.load_iris()
# getting the data and response of the dataset
x = irisdataset.data
y = irisdataset.target
# Creating the Model
model = LinearSVC()
model.fit(x, y)
#Predict by passing random values
print(model.predict([[1, 2, 3, 4]]))
# Accuracy is printed
print(model.score(x, y))