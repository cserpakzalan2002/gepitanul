import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def predict_subscription_status():
    st.title("Predict Subscription Status")

    st.write("Please provide the following information for prediction:")

    age = st.number_input("Age", min_value=18, value=30)
    gender = st.radio("Gender", ('Male', 'Female'))
    location = st.text_input("Location")
    payment_method = st.text_input("Payment Method")
    frequency = st.number_input("Frequency of Purchases", min_value=0, value=5)
    discount = st.number_input("Discount Applied", min_value=0, value=0)

    ok = st.button("Predict Subscription Status")

    df = pd.read_csv('shopping_trends_updated.csv')

    label_encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    X = df[['Age', 'Gender', 'Location', 'Payment Method', 'Frequency of Purchases', 'Discount Applied']]
    y = df['Subscription Status']

    model = RandomForestClassifier(n_estimators=100, random_state=0)

    if ok:
        model.fit(X, y)
        new_data = pd.DataFrame({
            'Age': [age],
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Location': label_encoders['Location'].transform([location])[0],
            'Payment Method': label_encoders['Payment Method'].transform([payment_method])[0],
            'Frequency of Purchases': [frequency],
            'Discount Applied': [discount]
        })

        predicted_status = model.predict(new_data)
        status = "Subscribed" if predicted_status[0] == 1 else "Not Subscribed"
        st.subheader(f"Estimated Subscription Status: {status}")

predict_subscription_status()
