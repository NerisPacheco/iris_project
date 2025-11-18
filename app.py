import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Configurar página
st.set_page_config(page_title="Clasificación Iris", layout="wide")

# Título
st.title("Clasificación de Especies Iris")
st.markdown("**Universidad de la Costa - Data Mining**")
st.markdown("---")

# Cargar datos
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['especie'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Sidebar
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["Analisis", "Modelo", "Predecir"])

if pagina == "Analisis":
    st.header("Analisis de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos del Dataset")
        st.write(f"Muestras: {len(df)}")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Distribucion de Especies")
        fig = px.pie(df, names='especie', title='Especies en el Dataset')
        st.plotly_chart(fig)
    
    st.subheader("Histograma por Caracteristica")
    caracteristica = st.selectbox("Selecciona:", iris.feature_names)
    fig2 = px.histogram(df, x=caracteristica, color='especie', title=f'Distribucion de {caracteristica}')
    st.plotly_chart(fig2)

elif pagina == "Modelo":
    st.header("Modelo de Machine Learning")
    
    # Entrenar modelo
    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Predecir y calcular metricas
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Precision del Modelo", f"{accuracy:.1%}")
    st.write(f"Muestras de entrenamiento: {len(X_train)}")
    st.write(f"Muestras de prueba: {len(X_test)}")
    
    # Matriz de confusion
    st.subheader("Matriz de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title="Matriz de Confusion")
    st.plotly_chart(fig_cm)

else:
    st.header("Predecir Especie")
    
    st.write("Ingresa las medidas de la flor:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepalo_largo = st.slider("Largo del Sepalo (cm)", 4.0, 8.0, 5.5)
        sepalo_ancho = st.slider("Ancho del Sepalo (cm)", 2.0, 4.5, 3.0)
    
    with col2:
        petalo_largo = st.slider("Largo del Petalo (cm)", 1.0, 7.0, 3.5)
        petalo_ancho = st.slider("Ancho del Petalo (cm)", 0.1, 2.5, 1.0)
    
    # Entrenar modelo
    X = df[iris.feature_names]
    y = df['species']
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    # Predecir
    medidas = [[sepalo_largo, sepalo_ancho, petalo_largo, petalo_ancho]]
    prediccion = modelo.predict(medidas)[0]
    probabilidades = modelo.predict_proba(medidas)[0]
    
    especies = ['Setosa', 'Versicolor', 'Virginica']
    especie_predicha = especies[prediccion]
    
    st.success(f"**Especie predicha: {especie_predicha}**")
    
    # Mostrar probabilidades
    st.subheader("Probabilidades:")
    for i, prob in enumerate(probabilidades):
        st.write(f"{especies[i]}: {prob:.1%}")

st.markdown("---")
st.markdown("Proyecto Final Data Mining - Universidad de la Costa")