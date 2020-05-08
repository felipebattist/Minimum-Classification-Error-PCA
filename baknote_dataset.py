import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from MY_pca import My_pca

n = 1

df = pd.read_csv('Banknote_dataset.csv')
data = pd.DataFrame(df)

target = data['class']

data.drop('class', inplace=True, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.25, random_state=0)
print('Sklearn PCA')
pca = PCA(n)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)

print(pca.explained_variance_)

model = svm.SVC(kernel='linear', C = 1.0)
model.fit(X_train1, Y_train)

sc = model.score(X_test1, Y_test)
print(sc)

print('My_pca_Bayes')

my_pca = My_pca(n)
data3 = my_pca.pca_bayes(data, target)
X_train2, X_test2, Y_train, Y_test = train_test_split(data3, target, train_size=0.25, random_state=0)

model3 = svm.SVC(kernel='linear', C = 1.0)
model3.fit(X_train2, Y_train)
sc = model3.score(X_test2, Y_test)
print(sc)