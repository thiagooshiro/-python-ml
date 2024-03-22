import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

#Objetivo criar um modelo que consiga dizer se uma temperatura é frio, confortável ou muito quente.
#Que seja capaz de classificar temperaturas para fora do trecho dado pelo dataframe.

#Dataframe de temperaturas e classificações:
file = pd.read_csv('tempe2.csv')

x, y = file['temperatures'].values, file['classification'].values

x= x.reshape(-1, 1)

## codificar classes não numéricas (pré-processamento)
encoder = LabelEncoder()

y = encoder.fit_transform(y.ravel())

#modelo escolhido
classifier = LogisticRegression()

#treinamento do modelo
classifier = classifier.fit(x, y)

#gerando 100 valores entre 0 e 45
x_test = np.linspace(start=0., stop=45., num=100)

# print(x_test)

#reformatan do o array para o formato necesário: -1 é "quantos valores existirem", 1 significa, organizado em 1 coluna
x_test = x_test.reshape(-1, 1)

# print(x_test)

#criando uma previsão baseada nos 100 valores de temperatura
y_pred = classifier.predict(x_test)


#utilizando codificaodr pra retornar os dados ao valor string deles
y_pred = encoder.inverse_transform(y_pred)

#criando um output considerando os dois valors:
output = {
    "new_temp": x_test.ravel(),
    "new_class": y_pred.ravel() 
}

output = pd.DataFrame(output)

# output.head()
 
print(output.describe())
 
output["new_class"].value_counts().plot().bar(figsize=0.5, rot=0, title='valores gerados')

output.boxplot(by="new_ class",  figsize=(10, 5))