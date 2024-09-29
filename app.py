import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
st.title("Random Forest Classifier - Social Network Ads")
st.write("Model ini memprediksi apakah seseorang akan membeli produk berdasarkan Usia dan Gaji")


# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Preprocessing
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Sidebar for user input
st.sidebar.header('Input Features')
age = st.sidebar.slider('Age', min_value=18, max_value=60, value=30)
salary = st.sidebar.slider('Estimated Salary', min_value=15000, max_value=150000, value=50000)

# Predict button
if st.sidebar.button('Predict'):
    user_data = scaler.transform([[age, salary]])
    prediction = rf.predict(user_data)
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Prediction: Will the user purchase? {result}")

# Show dataset
st.write("Dataset Sample")
st.write(df.head())
