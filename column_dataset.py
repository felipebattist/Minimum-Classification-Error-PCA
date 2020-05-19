import pandas as pd
import func_auxiliares as fc
n = 6
print('VColumn Dataset - n_features: '+str(n))

df = pd.read_csv('datasets/column.csv')
data = pd.DataFrame(df)
target = data['class']
data.drop('class', inplace=True, axis=1)


x = []
y = []
for i in range(1,n+1):
    a = fc.naive_pca(data, target, i)
    b = fc.naive_proposed(data, target, i)
    x.append(a)
    y.append(b)

print('Sklearn PCA')
teste = list(map(str, x))
print(' '.join(teste))
print('Proposed PCA')
teste = list(map(str, y))
print(' '.join(teste))