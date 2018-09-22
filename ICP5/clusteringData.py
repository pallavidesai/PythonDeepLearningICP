#  Creating linear regression model for the dataset using NumPy.
#  And Ploting the model using matplotlib.
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_url = 'train1.csv'
train = pd.read_csv(train_url)

#print(train.isnull().head())
# Fill missing values with mean column values in the test set
train.fillna(train.mean(), inplace=True)

#Survival based on the X---> the mean is greater for women
train[["X", "Y"]].groupby(['X'], as_index=False).mean().sort_values(by='Y', ascending=False)

#lets convert X which is nominal to numerical value :D
labelencoding = LabelEncoder()
labelencoding.fit(train['X'])
train['X'] = labelencoding.transform(train['X'])
# Drop subject colomn since we are not calculating for that
X=np.array(train.drop(['subject'],axis=1))
# Since we have to create 2 clusters and lets iterate for 25 times.
kmeans = KMeans(n_clusters=2,max_iter=100, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
import matplotlib.pyplot as plt


