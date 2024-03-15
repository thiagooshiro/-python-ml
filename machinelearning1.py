from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

regression = LinearRegression(fit_intercept=True)
file = pd.read_csv('students100.csv')
columns = ['Idade', 'Faltas', 'Matemática', 'Ciências', 'História']
x = file.loc[:, 'Matemática'].values
y = file.loc[:, 'Horas de estudo'].values

x = x.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

regression.fit(x_test, y_test)
y_pred = regression.predict(x_train)

plt.scatter(x_train, y_train, color='green', label='Dados Treino')
plt.scatter(x_test, y_test, color='red', label='Dados Teste')
plt.plot(x_train, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.title('Horas de Estudo/Desempenho')
plt.xlabel('Tempo em horas')
plt.ylabel('Nota em Matemática')
plt.legend()

print(regression.score(x_train, y_train))