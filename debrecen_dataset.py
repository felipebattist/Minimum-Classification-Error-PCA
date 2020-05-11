import pandas as pd
import func_auxiliares as fc

n = 3
print('Debrecen Dataset - n_features: '+str(n))
df = pd.read_csv('debrecen.csv')
data = pd.DataFrame(df)
target = data['class']
data.drop('class', inplace=True, axis=1)

fc.svm_pca_test(data, target, n)
fc.svm_pcaBayes_teste(data, target, n)

