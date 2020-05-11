import pandas as pd
import func_auxiliares as fc
n = 11
print('Climate Dataset - n_features: '+str(n))

df = pd.read_csv('climate.csv')
data = pd.DataFrame(df)
target = data['outcome']
data.drop('outcome', inplace=True, axis=1)

fc.svm_pca_test(data, target, n)
fc.svm_pcaBayes_teste(data, target, n)