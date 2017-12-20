import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
#loaded CSV file and dropped unnecessary files
men = pd.read_csv("C:/Users/tankn/PycharmProjects/cpsc483/AusOpen-men-2013.csv")
men.head
men = men.drop(["Player1","Player2"],axis=1)
men.shape
mens = men.drop("Result", axis=1)

y = men["Result"].copy()
# split data into test and training
X_train, X_test, y_train, y_test = train_test_split(mens, y)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
imputer = Imputer(strategy="median")
X_train = imputer.fit_transform(X_train)

#applied GaussianNB classifier to Online_Retail Dataset
clf = GaussianNB()
clf.fit(X_train,y_train)
X_test = imputer.fit_transform(X_test)
pfm =clf.predict(X_test)
print(pfm)
#print(classification_report(pfm,y_test))
print("performance of GaussianNB classifier",np.mean(pfm == y_test))

#applying KNeighborsClassifier
dp = int(np.sqrt(len(X_train)))

for i in range(5,dp):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    pre = neigh.predict(X_test)
    err = metrics.zero_one_loss(y_test,pre)
    print("Error at",i,"is",err)

print("performance of KNeighborsClassifier",np.mean(pre == y_test))

#SVC
clf1 = SVC()
clf1.fit(X_train, y_train)
#poly kernal and degree 5
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=5, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svc = clf.predict(X_test)
print("Predicted value",svc)
print("Error",metrics.zero_one_loss(y_test,svc))
print("performance of SVC(poly) at degree 5",np.mean(svc == y_test))

#poly kernal and degree 3
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svcp3 = clf.predict(X_test)
print("Predicted value",svcp3)
print("Error",metrics.zero_one_loss(y_test,svcp3))
print("performance of SVC(poly) at degree 3",np.mean(svcp3 == y_test))

#linear kernal and degree 5
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=5, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svcl5 = clf.predict(X_test)
print("predicted Value",svcl5)
print("Error",metrics.zero_one_loss(y_test,svcl5))
print("performance of SVC(linear)",np.mean(svcl5 == y_test))

#linear kernal and degree 3
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
svcl3 = clf.predict(X_test)
print("predicted Value",svcl3)
print("Error",metrics.zero_one_loss(y_test,svcl3))
print("performance of SVC(linear)",np.mean(svcl3 == y_test))

#clustering
cdata = pd.read_csv("C:/Users/tankn/PycharmProjects/cpsc483/AusOpen-men-2013cluster1.csv")
kmeans = KMeans(n_clusters=2).fit(cdata)
print(kmeans.predict([y_test]))