import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

# Cargar los datos
bitcoin_data = pd.read_csv('bitcoin.csv')
bitcoin_data['datetime'] = pd.to_datetime(bitcoin_data['datetime'], format='%d/%m/%y')
bitcoin_data_prophet = bitcoin_data[['datetime', 'Close']].rename(columns={'datetime': 'ds', 'Close': 'y'})

# Entrenar modelo Prophet
model = Prophet()
model.fit(bitcoin_data_prophet)
forecast = model.predict(bitcoin_data_prophet)

# Mostrar R²
y_real = bitcoin_data_prophet['y']
y_pred = forecast['yhat']
r2 = r2_score(y_real, y_pred)
st.markdown(f"### R² del modelo: `{r2:.4f}`")

# Sidebar: selección de filtros
st.sidebar.title("Filtros de Visualización")
opcion = st.sidebar.selectbox("Selecciona periodo", ["Año", "Mes", "Semana", "Rango de Fechas"])

año = st.sidebar.selectbox("Año", [2021, 2022, 2023, 2024, 2025])
mes = st.sidebar.selectbox("Mes", list(range(1, 13)))
semana = st.sidebar.slider("Semana", 1, 52)

# Determinar rango seleccionado
fecha_inicio = None
fecha_fin = None

if opcion == 'Año':
    fecha_inicio = datetime(año, 1, 1)
    fecha_fin = datetime(año, 12, 31)
elif opcion == 'Mes':
    fecha_inicio = datetime(año, mes, 1)
    fecha_fin = datetime(año, mes, pd.Period(year=año, month=mes, freq='M').days_in_month)
elif opcion == 'Semana':
    fecha_inicio = datetime(año, 1, 1) + timedelta(weeks=semana-1)
    fecha_inicio -= timedelta(days=fecha_inicio.weekday())  # lunes
    fecha_fin = fecha_inicio + timedelta(days=6)
elif opcion == 'Rango de Fechas':
    fecha_inicio = pd.to_datetime(st.sidebar.date_input("Fecha Inicio", datetime(2023, 1, 1), key="fecha_inicio"))
    fecha_fin = pd.to_datetime(st.sidebar.date_input("Fecha Fin", datetime(2023, 12, 31), key="fecha_fin"))

# Filtrado de datos
def filtrar_datos(periodo, fecha_inicio=None, fecha_fin=None):
    if periodo == 'Año':
        return bitcoin_data[(bitcoin_data['datetime'].dt.year == fecha_inicio.year) & (bitcoin_data['datetime'] <= fecha_fin)]
    elif periodo == 'Mes':
        return bitcoin_data[(bitcoin_data['datetime'].dt.year == fecha_inicio.year) & (bitcoin_data['datetime'].dt.month == fecha_inicio.month)]
    elif periodo == 'Semana':
        return bitcoin_data[(bitcoin_data['datetime'].dt.year == fecha_inicio.year) & (bitcoin_data['datetime'].dt.isocalendar().week == fecha_inicio.isocalendar().week)]
    elif periodo == 'Rango de Fechas':
        return bitcoin_data[(bitcoin_data['datetime'] >= fecha_inicio) & (bitcoin_data['datetime'] <= fecha_fin)]
    return bitcoin_data

# Gráfica
bitcoin_filtrado = filtrar_datos(opcion, fecha_inicio, fecha_fin)

st.subheader("Gráfica de Precios")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bitcoin_filtrado['datetime'], bitcoin_filtrado['Close'], marker='o')
ax.set_title("Precio de Bitcoin")
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio de Cierre")
plt.xticks(rotation=45)
st.pyplot(fig)

# Mostrar últimos 7 días
st.subheader("Últimos 7 días")
st.table(bitcoin_data.tail(7)[['datetime', 'Close']].rename(columns={'datetime': 'Fecha', 'Close': 'Precio Cierre'}))

# Predicción de semana futura
st.subheader("Predicción para Semana Futura")

año_pred = st.number_input("Año (mínimo 2025)", min_value=2025, step=1)
semana_pred = st.number_input("Semana", min_value=1, max_value=52, step=1)

if año_pred == 2025 and semana_pred < 18:
    st.warning("La semana debe ser 18 o superior para 2025.")
else:
    if st.button("Predecir Semana"):
        fecha_inicio = datetime(año_pred, 1, 1) + timedelta(weeks=semana_pred-1)
        fecha_inicio -= timedelta(days=fecha_inicio.weekday())
        fechas = [fecha_inicio + timedelta(days=i) for i in range(7)]
        future = pd.DataFrame({'ds': fechas})
        predicciones = model.predict(future)[['ds', 'yhat']]
        st.table(predicciones.rename(columns={'ds': 'Fecha', 'yhat': 'Precio Predicho'}))
