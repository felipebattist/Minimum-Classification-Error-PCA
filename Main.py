import pandas as pd
import matplotlib.pyplot as plt
import func_auxiliares as fc


# classifcadores : KNN, naive_bayes, SVM, LDA, decisiontree
classificador = 'DecisionTree'
Datasets = {'column':'column.csv', 'banknote': 'Banknote_dataset.csv',  'climate': 'climate.csv', 'debrecen': 'debrecen.csv', 'spambase': 'spambase.csv'}


dataset = Datasets['debrecen']


print('Classificador: ' + classificador)

df = pd.read_csv('datasets/' + dataset)
data = pd.DataFrame(df)
target = data['class']
data.drop('class', inplace=True, axis=1)
n = data.shape[1]
print(dataset + ' - n_features: ' + str(n))
pca, proposed = fc.select_classifier(n, data, target, classificador)

eixo_x = []
for j in range(len(proposed)):
    eixo_x.append(j+1)

print('Sklearn PCA')
teste = list(map(str, pca))
print(' '.join(teste))
print('Proposed PCA')
teste = list(map(str, proposed))
print(' '.join(teste))

t = dataset + ' - Classificador: ' + classificador
plt.plot(eixo_x, proposed, label = 'Proposed', color = 'blue')
plt.scatter(eixo_x, proposed, zorder = 1)
plt.plot(eixo_x, pca, label = 'PCA', color = 'green')
plt.scatter(eixo_x, pca, zorder = 1, color = 'green')
plt.title(t)
plt.legend()
plt.grid()
plt.show()
