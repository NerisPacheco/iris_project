import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="Iris Classification", layout="wide")

# Title
st.title("Iris Species Classification")
st.markdown("**University of Costa - Data Mining**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
    return df, iris

try:
    df, iris = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Data Analysis", "Machine Learning Model", "Predict Species"])
    
    # PAGE 1: DATA ANALYSIS
    if page == "Data Analysis":
        st.header("📊 Data Analysis and Understanding")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(iris.feature_names))
        with col3:
            st.metric("Species", len(df['species_name'].unique()))
        
        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Species distribution
        st.subheader("Species Distribution")
        col1, col2 = st.columns(2)
        with col1:
            species_count = df['species_name'].value_counts()
            fig_pie = px.pie(values=species_count.values, names=species_count.index, 
                           title="Species Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(species_count, x=species_count.index, y=species_count.values,
                           title="Species Count", labels={'x': 'Species', 'y': 'Count'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions by Species")
        feature = st.selectbox("Select feature to visualize:", iris.feature_names)
        fig_hist = px.histogram(df, x=feature, color='species_name', nbins=20,
                              title=f"Distribution of {feature} by Species")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Scatter plot
        st.subheader("Feature Relationships")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature:", iris.feature_names, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis feature:", iris.feature_names, index=1)
        
        fig_scatter = px.scatter(df, x=x_feature, y=y_feature, color='species_name',
                               title=f"{y_feature} vs {x_feature}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        numeric_df = df[iris.feature_names]
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # PAGE 2: MACHINE LEARNING MODEL
    elif page == "Machine Learning Model":
        st.header("🤖 Machine Learning Model")
        st.write("**Algorithm: Random Forest Classifier**")
        
        # Train-test split
        X = df[iris.feature_names]
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        with col4:
            st.metric("F1-Score", f"{f1:.2%}")
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Setosa', 'Versicolor', 'Virginica'],
                          y=['Setosa', 'Versicolor', 'Virginica'],
                          title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature', 
                              orientation='h', title="Feature Importance in Random Forest")
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # PAGE 3: PREDICT SPECIES
    else:
        st.header("🔮 Predict Iris Species")
        st.write("Enter the flower measurements to predict the species:")
        
        # Input sliders
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        with col2:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0, 0.1)
        
        # Train model on all data
        X = df[iris.feature_names]
        y = df['species']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Make prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        predicted_species = species_names[prediction]
        
        # Display prediction
        st.success(f"**Predicted Species: {predicted_species}**")
        
        # Probabilities
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Species': species_names,
            'Probability': probabilities
        })
        
        fig_prob = px.bar(prob_df, x='Species', y='Probability', color='Probability',
                         text_auto=True, range_y=[0, 1],
                         title="Species Prediction Probabilities")
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # 3D Scatter Plot with new sample
        st.subheader("3D Visualization with New Sample")
        fig_3d = px.scatter_3d(df, x='sepal length (cm)', y='sepal width (cm)', 
                              z='petal length (cm)', color='species_name',
                              title="3D Distribution of Iris Species")
        
        # Add new sample to the plot
        fig_3d.add_trace(go.Scatter3d(
            x=[sepal_length], y=[sepal_width], z=[petal_length],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Your Sample'
        ))
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Show input values
        st.subheader("Your Input Values")
        input_df = pd.DataFrame({
            'Feature': iris.feature_names,
            'Value': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df)

except Exception as e:
    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Final Data Mining Project - University of Costa**")
st.markdown("Department of Computer Science and Electronics")
