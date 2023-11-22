import streamlit as st
import joblib 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from page import streamlitShow




import streamlit as st
import joblib
import numpy as np

def predict_item_brand(name, generation, color, number_of_sim):
    # Betöltjük a Decision Tree modelled
    decision_tree_model = joblib.load('decision_tree_model.joblib')  # Ezt cseréld le a valós fájlnevre

    # Ellenőrizzük az input adatokat és átalakítjuk numpy tömbbé
    input_data = np.array([name, generation, color, number_of_sim]).reshape(1, -1)

    # Predikció a Decision Tree modelled alapján
    prediction = decision_tree_model.predict(input_data)

    return prediction[0]  # Visszaadjuk a predikciót

def main():
    st.title('Elem Márka Predikció')

    name = st.text_input("Termék neve")
    generation = st.text_input("Termék generációja")
    color = st.text_input("Termék színe")
    number_of_sim = st.number_input("SIM kártyák száma", min_value=0, step=1)

    if st.button("Predikció"):
        result = predict_item_brand(name, generation, color, number_of_sim)
        st.write(f"Az elem márka predikciója: {result}")

if __name__ == "__main__":
    main()







