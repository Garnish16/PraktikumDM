import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Judul aplikasi
st.title("Prediksi Konsumsi BBM Berdasarkan Jarak Tempuh (Regresi Linier)")

# Data latih (hardcoded)
data_latih = {
    'jarak_tempuh': [10, 15, 20, 25, 30, 35, 40],
    'total_konsumsi': [1.2, 1.7, 2.1, 2.6, 3.0, 3.5, 4.1]
}
data = pd.DataFrame(data_latih)

st.subheader("Data Latih Konsumsi BBM")
st.dataframe(data)

# Training model regresi linier
X = data[['jarak_tempuh']]
y = data['total_konsumsi']
model = LinearRegression()
model.fit(X, y)

# Input user
st.subheader("Masukkan Jarak Tempuh (km)")
jarak_input = st.number_input("Jarak Tempuh", min_value=0.0, step=0.1)

if st.button("Prediksi Konsumsi BBM"):
    prediksi = model.predict([[jarak_input]])
    st.success(f"Prediksi konsumsi BBM untuk {jarak_input:.1f} km: {prediksi[0]:.2f} liter")

    # Visualisasi regresi
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Data Latih')
    plt.plot(x_range, y_pred, color='red', label='Model Regresi')
    plt.xlabel("Jarak Tempuh (km)")
    plt.ylabel("Konsumsi BBM (liter)")
    plt.title("Regresi Linier Konsumsi BBM")
    plt.legend()
    st.pyplot(plt)
