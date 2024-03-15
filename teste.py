import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# Dados de exemplo
pesos = np.array([65, 70, 80, 85, 90])
alturas = np.array([160, 165, 175, 180, 185])
# Organizando os dados
x = pesos.reshape(-1, 1)
y = alturas
# Criando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x, y)
# Fazendo as previsões
y_pred = modelo.predict(x)
# Plotando o gráfico
plt.scatter(x, y, color='blue', label='Dados Originais')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.title('Relação: Peso e Altura')
plt.xlabel('Peso em Kg')
plt.ylabel('Altura em cm')
plt.legend()
