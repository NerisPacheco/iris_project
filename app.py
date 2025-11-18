import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

st.title("Iris Dataset Test")
st.write("Testing if data loads correctly...")

# Try to load data
try:
    iris = load_iris()
    st.success("✅ Iris dataset loaded successfully!")
    
    # Show basic info
    st.write(f"Number of samples: {iris.data.shape[0]}")
    st.write(f"Number of features: {iris.data.shape[1]}")
    st.write(f"Feature names: {iris.feature_names}")
    st.write(f"Target names: {iris.target_names}")
    
    # Create dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    st.subheader("First 5 rows of data:")
    st.dataframe(df.head())
    
    st.subheader("Basic statistics:")
    st.dataframe(df.describe())
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
