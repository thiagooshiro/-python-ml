import pandas as pd

data = [10, 20, 30, 40, 50] # Substitua isso pelos seus próprios dados

start = '2023-01-01' # Substitua isso pela sua data de início
end = '2023-01-05' # Substitua isso pela sua data de término
frequency = 'D' # Substitua isso pela sua frequência (D para dias, M para meses, etc.)

timeseries = pd.Series(data, index=pd.date_range(start, end, freq=frequency))
# Exibindo a série temporal
print(timeseries)