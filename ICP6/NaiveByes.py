# Implementing Naïve Bayes method using scikit-learn library
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
# Loading the dataset
irisdataset = datasets.load_iris()
x = irisdataset.data
y = irisdataset.target
# getting the data and response of the dataset
# For doing cross validation we will keep 20% for test and 80% for training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Creating the Model
model = GaussianNB()
model.fit(x_train, y_train)
# Do cross validation now
y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

