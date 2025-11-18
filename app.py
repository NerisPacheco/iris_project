import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Configure page
st.set_page_config(page_title="Iris Classification", layout="wide")

# Title
st.title("Iris Species Classification")
st.markdown("**University of Costa - Data Mining**")
st.markdown("---")

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Analysis", "ML Model", "Predict Species"])

if page == "Data Analysis":
    st.header("Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset")
        st.write(f"Samples: {len(df)}")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Species Distribution")
        fig = px.pie(df, names='species_name', title='Species in Dataset')
        st.plotly_chart(fig)
    
    st.subheader("Feature Histogram")
    feature = st.selectbox("Select feature:", iris.feature_names)
    fig2 = px.histogram(df, x=feature, color='species_name', title=f'Distribution of {feature}')
    st.plotly_chart(fig2)

elif page == "ML Model":
    st.header("Machine Learning Model")
    
    # Train model
    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy", f"{accuracy:.1%}")
    st.write(f"Training samples: {len(X_train)}")
    st.write(f"Test samples: {len(X_test)}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig_cm)

else:
    st.header("Predict Species")
    
    st.write("Enter flower measurements:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    
    # Train model
    X = df[iris.feature_names]
    y = df['species']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict
    measurements = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(measurements)[0]
    probabilities = model.predict_proba(measurements)[0]
    
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction]
    
    st.success(f"**Predicted species: {predicted_species}**")
    
    # Show probabilities
    st.subheader("Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"{species_names[i]}: {prob:.1%}")

st.markdown("---")
st.markdown("Final Project - Data Mining - University of Costa")
