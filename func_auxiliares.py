import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from MY_pca import My_pca

def svm_pca_test(data, target, n):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.35, random_state=0)
    pca = PCA(n)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)
    model = svm.SVC(kernel='linear', C = 0.001)
    model.fit(X_train1, Y_train)
    sc = model.score(X_test1, Y_test)
    return sc


def svm_pca_proposed(data, target, n):
    my_pca = My_pca(n)
    data3 = my_pca.MCPCA(data, target)
    X_train2, X_test2, Y_train, Y_test = train_test_split(data3, target, train_size=0.35, random_state=0)

    model3 = svm.SVC(kernel='linear', C = 0.001)
    model3.fit(X_train2, Y_train)
    sc = model3.score(X_test2, Y_test)
    return sc


def knn_pca(data, target, n):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.35, random_state=0)
    pca = PCA(n)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train1, Y_train)
    y_expect = Y_test
    y_pred = knn.predict(X_test1)
    return (metrics.accuracy_score(y_expect, y_pred))

def knn_proposed(data, target, n):
    my_pca = My_pca(n)
    data3 = my_pca.MCPCA(data, target)
    X_train2, X_test2, Y_train, Y_test = train_test_split(data3, target, train_size=0.35, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train2, Y_train)
    y_expect = Y_test
    y_pred = knn.predict(X_test2)
    return (metrics.accuracy_score(y_expect, y_pred))


def decisionTree_pca(data, target, n):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.35, random_state=0)
    pca = PCA(n)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train1, Y_train)
    return (clf.score(X_test1, Y_test))

def decisionTree_proposed(data, target, n):
    my_pca = My_pca(n)
    data3 = my_pca.MCPCA(data, target)
    X_train2, X_test2, Y_train, Y_test = train_test_split(data3, target, train_size=0.35, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train2, Y_train)
    return (clf.score(X_test2, Y_test))

def naive_pca(data, target, n):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.35, random_state=0)
    pca = PCA(n)
    X_train1 = pca.fit_transform(X_train)
    X_test1 = pca.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train1,Y_train)
    return clf.score(X_test1, Y_test)

def naive_proposed(data, target, n):
    my_pca = My_pca(n)
    data3 = my_pca.MCPCA(data, target)
    X_train2, X_test2, Y_train, Y_test = train_test_split(data3, target, train_size=0.35, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train2, Y_train)
    return clf.score(X_test2, Y_test)






