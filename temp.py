import streamlit as st
import pandas as pd
import joblib

st.write("# Depression Level Prediction")

col1, col2, col3 = st.columns(3)

# Getting user input
phq1 = col1.selectbox("1. Little interest or pleasure in doing things", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq2 = col2.selectbox("2. Feeling down, depressed, or hopeless", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq3 = col3.selectbox("3. Trouble sleeping or sleeping too much", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq4 = col1.selectbox("4. Feeling tired or having little energy", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq5 = col2.selectbox("5. Poor appetite or overeating", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq6 = col3.selectbox("6. Feeling failure, bad about yourself", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq7 = col1.selectbox("7. Trouble concentrating on things, reading, watching TV", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq8 = col2.selectbox("8. Moving or speaking slowly or being restless", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
phq9 = col3.selectbox("9. Thoughts of self-harm or suicide", ["Not at all", "Several days", "More than half the days", "Nearly every day"])

btn = st.button('Predict', type="primary")

df_pred = pd.DataFrame([[phq1, phq2, phq3, phq4, phq5, phq6, phq7, phq8, phq9]], 
                       columns=['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9'])

def transform(data):
    result = 0
    if data == "Several days":
        result = 1
    elif data == "More than half the days":
        result = 2
    elif data == "Nearly every day":
        result = 3
    return result

for col in df_pred.columns:
    df_pred[col] = df_pred[col].apply(transform)

# Load the saved model for prediction
    
    mlp_model = joblib.load(r"C:\Users\kashi\OneDrive\Desktop\Research Paper\mlp_model.pkl")

prediction = mlp_model.predict(df_pred)

if btn:
    st.write('<b>Your Depression Level: </b>', prediction, unsafe_allow_html=True)