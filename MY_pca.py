import numpy as np
from numpy import array
#Define um dataset em formato de matriz



class My_pca:
    def __init__(self, n_components = 0):
        self.n_components = n_components

    def prodMatrix(self,matrizA, matrizB):
        """Multiplica duas matrizes."""
        sizeLA = len(matrizA)
        sizeCA = len(matrizA[0])  # deve ser igual a sizeLB para ser possível multiplicar as matrizes
        sizeCB = len(matrizB[0])
        matrizR = []
        # Multiplica
        for i in range(sizeLA):
            matrizR.append([])
            for j in range(sizeCB):
                val = 0
                for k in range(sizeCA):
                    val += matrizA[i][k] * matrizB[k][j]
                matrizR[i].append(val)
        return matrizR

    def max_pos(self, valores):
        maior = valores[0]
        maior_pos = 0
        for i in range(1,len(valores)):
            if valores[i] > maior:
                maior = valores[i]
                maior_pos = i

        return maior_pos, maior

    def get_pca(self, Dataset):
        n_components = self.n_components
        # Cria uma matriz com a media de cada coluna.
        M = np.mean(Dataset.T, axis=1)
        # Centraliza as colunas do dataset por subtrair as medias.
        C = Dataset - M
        # Calcula a matriz de covariancia da matriz centralizada.
        COV = np.cov(C.T)
        # Gera os auto-valores e autovetores da matriz de covariancia.
        valores, vetores = np.linalg.eig(COV)
        valores = valores.tolist()
        P = vetores.T.dot(C.T)
        P = (P.T).tolist()
        if n_components == 0 or n_components == len(valores):
            print(valores)
            return (P)
        elif n_components > 0 and n_components < len(valores):
            new_vetores = []
            new_valores = []
            size = len(P)
            for k in range(size):
                new_vetores.append([])
            for n in range(n_components):
                pos, val = self.max_pos(valores)
                new_valores.append(val)
                valores[pos] = -1000
                for i in range(size):
                    new_vetores[i].append(P[i][pos])
            print(new_valores)
            return new_vetores

        else:
            print('ERRO: n_components must be >= 0')


    def class_appearance(self, target):
        #Retorna o núumero de aparições das classes c1 e c2
        c1 = 0
        c2 = 0
        x = target[0]
        for i in range(len(target)):
            if x == target[i]:
                c1 +=1
            else:
                c2 +=1
        return c1, c2

    def feature_mean(self,n_features, data, target):
        #Retorna a media de cada feature para as classes c1 e c2
        target = target.tolist()
        size = len(target)
        x = target[0]
        c1 , c2 = self.class_appearance(target)
        c1_medias = []
        c2_medias = []
        for i in range (n_features):
            media1 = 0
            media2 = 0
            for j in range(size):
                if(target[j] == x):
                    media1 += data[j][i]
                else:
                    media2 += data[j][i]

            media1 = (media1)/c1
            media2 = (media2)/c2
            c1_medias.append(media1)
            c2_medias.append(media2)
        return c1_medias, c2_medias

    def score_bayes(self, media1, media2, auto_valores):
        #Retorna um vetor com os scores de cada auto valor.
        scores = []
        for i in range(len(media1)):
            if auto_valores[i] != 0:
                sc = ((media1[i] - media2[i])**2)/auto_valores[i]
                scores.append(sc)
            else:
                scores.append(0)
        return scores

    def MCPCA(self, data, target):
        n_components = self.n_components

        # Cria uma matriz com a media de cada coluna.
        M = np.mean(data.T, axis=1)
        # Centraliza as colunas do dataset por subtrair as medias.
        C = data - M
        # Calcula a matriz de covariancia da matriz centralizada.
        COV = np.cov(C.T)
        # Gera os auto-valores e autovetores da matriz de covariancia.
        valores, vetores = np.linalg.eig(COV)
        valores = valores.tolist()
        P = vetores.T.dot(C.T)
        P = (P.T).tolist()
        size = len(P)
        if n_components == 0 or n_components == len(valores):
            return (P)
        elif n_components > 0 and n_components < len(valores):
            media1, media2 = self.feature_mean(len(valores), P, target)
            scores = self.score_bayes(media1, media2, valores)
            new_vetores = []
            new_valores = []
            for k in range(size):
                new_vetores.append([])
            for n in range(n_components):
                pos, val = self.max_pos(scores)
                new_valores.append(val)
                scores[pos] = -1000
                for i in range(size):
                    new_vetores[i].append(P[i][pos])

            results = np.array(new_vetores)
            return results


        else:
            print('ERRO: n_components must be >= 0')