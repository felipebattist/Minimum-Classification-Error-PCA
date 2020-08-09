import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from MY_pca import My_pca
from sklearn import discriminant_analysis

def LDA_pca(data, target, n):
    pca = PCA(n)
    data1 = pca.fit_transform(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data1, target, train_size=0.35, random_state=0)

    model = discriminant_analysis.LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    sc = model.score(X_test, Y_test)
    return sc

def LDA_proposed(data, target, n):
    my_pca = My_pca(n)
    results = my_pca.MCPCA_fit_transform(data, target)
    X_train, X_test, Y_train, Y_test = train_test_split(results['new_data'], target, train_size=0.35, random_state=0)

    model = discriminant_analysis.LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    sc = model.score(X_test, Y_test)
    return sc


def svm_pca_test(data, target, n):
    pca = PCA(n)
    data1 = pca.fit_transform(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data1, target, train_size=0.35, random_state=0)
    model = svm.SVC(kernel='linear', C = 0.8)
    model.fit(X_train, Y_train)
    sc = model.score(X_test, Y_test)
    return sc


def svm_pca_proposed(data, target, n):
    my_pca = My_pca(n)
    results = my_pca.MCPCA_fit_transform(data, target)
    X_train, X_test, Y_train, Y_test = train_test_split(results['new_data'], target, train_size=0.35, random_state=0)

    model3 = svm.SVC(kernel='linear', C = 0.8)
    model3.fit(X_train, Y_train)
    sc = model3.score(X_test, Y_test)
    return sc


def knn_pca(data, target, n):
    pca = PCA(n)
    data1 = pca.fit_transform(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data1, target, train_size=0.35, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, Y_train)
    y_expect = Y_test
    y_pred = knn.predict(X_test)
    return (metrics.accuracy_score(y_expect, y_pred))

def knn_proposed(data, target, n):
    my_pca = My_pca(n)
    results = my_pca.MCPCA_fit_transform(data, target)
    X_train, X_test, Y_train, Y_test = train_test_split(results['new_data'], target, train_size=0.35, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, Y_train)
    y_expect = Y_test
    y_pred = knn.predict(X_test)
    return (metrics.accuracy_score(y_expect, y_pred))


def decisionTree_pca(data, target, n):
    pca = PCA(n)
    data1 = pca.fit_transform(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data1, target, train_size=0.35, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, Y_train)
    return (clf.score(X_test, Y_test))

def decisionTree_proposed(data, target, n):
    my_pca = My_pca(n)
    results = my_pca.MCPCA_fit_transform(data, target)
    X_train, X_test, Y_train, Y_test = train_test_split(results['new_data'], target, train_size=0.35, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, Y_train)
    return (clf.score(X_test, Y_test))

def naive_pca(data, target, n):
    pca = PCA(n)
    data1 = pca.fit_transform(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data1, target, train_size=0.35, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    return clf.score(X_test, Y_test)

def naive_proposed(data, target, n):
    my_pca = My_pca(n)
    results = my_pca.MCPCA_fit_transform(data, target)
    X_train, X_test, Y_train, Y_test = train_test_split(results['new_data'], target, train_size=0.35, random_state=0)

    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    return clf.score(X_test, Y_test)


def select_classifier(n, data, target,classificador = 'nda'):
    pca = []
    proposed = []
    if classificador == 'naive_bayes':
        for i in range(1, n + 1):
            a = naive_pca(data, target, i)
            b = naive_proposed(data, target, i)
            pca.append(a)
            proposed.append(b)
    elif classificador == 'KNN':
        for i in range(1, n + 1):
            a = knn_pca(data, target, i)
            b = knn_proposed(data, target, i)
            pca.append(a)
            proposed.append(b)
    elif classificador == 'SVM':
        for i in range(1, n + 1):
            a = svm_pca_test(data, target, i)
            b = svm_pca_proposed(data, target, i)
            pca.append(a)
            proposed.append(b)

    elif classificador == 'LDA':
        for i in range(1, n + 1):
            a = LDA_pca(data, target, i)
            b = LDA_proposed(data, target, i)
            pca.append(a)
            proposed.append(b)
    else:
        for i in range(1, n + 1):
            a = decisionTree_pca(data, target, i)
            b = decisionTree_proposed(data, target, i)
            pca.append(a)
            proposed.append(b)

    return pca, proposed






