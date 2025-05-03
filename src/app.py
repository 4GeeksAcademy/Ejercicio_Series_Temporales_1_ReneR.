from utils import db_connect
engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# Cargar y preparar el dataset
df = pd.read_csv('Series de tiempo.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
daily_sales = df.resample('D').mean()

# Visualización de la serie temporal
plt.figure(figsize=(14, 6))
sns.lineplot(data=daily_sales, x=daily_sales.index, y='sales')
plt.title('Serie Temporal de Ventas Diarias')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.grid(True)
plt.tight_layout()
plt.show()

# Prueba de estacionariedad (ADF)
result = adfuller(daily_sales['sales'].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
for key, value in result[4].items():
    print(f"Valor crítico {key}: {value}")

# Entrenamiento del modelo ARIMA
model = ARIMA(daily_sales['sales'], order=(1, 1, 1))
fitted_model = model.fit()
print(fitted_model.summary())

# División en train-test
train = daily_sales[:-30]
test = daily_sales[-30:]

# Reentrenar sobre train
model = ARIMA(train['sales'], order=(1, 1, 1))
fitted_model = model.fit()

# Predicción
forecast = fitted_model.forecast(steps=30)
forecast.index = test.index

# Visualización de predicción vs. real
plt.figure(figsize=(12, 5))
plt.plot(train.index, train['sales'], label='Entrenamiento')
plt.plot(test.index, test['sales'], label='Real', color='green')
plt.plot(forecast.index, forecast, label='Predicción', color='red', linestyle='--')
plt.title('Predicción vs. Valores Reales')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Métricas de evaluación
mae = mean_absolute_error(test['sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Guardar modelo
joblib.dump(fitted_model, 'modelo_arima_ventas.pkl')
