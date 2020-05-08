import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from MY_pca import My_pca

def svm_test(X_train, X_test, Y_train, Y_test):

    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X_train, Y_train)

    sc = model.score(X_test, Y_test)
    print(sc)


n = 1
print('Banknote Dataset - n_features: '+str(n))

df = pd.read_csv('Banknote_dataset.csv')
data = pd.DataFrame(df)

target = data['class']

data.drop('class', inplace=True, axis=1)


print('Sklearn PCA')

X_train, X_test, Y_train1, Y_test1 = train_test_split(data, target, train_size=0.25, random_state=0)
pca = PCA(n)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)


svm_test(X_train1, X_test1,Y_train1,Y_test1)

print('My_pca_Bayes')

my_pca = My_pca(n)
data1 = my_pca.pca_bayes(data, target)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(data1, target, train_size=0.25, random_state=0)

svm_test(X_train2,X_test2, Y_train2, Y_test2)