# app.py
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "income_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    trained = True
else:
    df = pd.read_csv("adult 3.csv")
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    selected_features = ["age", "education", "marital-status", "occupation", "hours-per-week", "gender"]
    target_column = "income"
    categorical_cols = ["education", "marital-status", "occupation", "gender", "income"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[selected_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    trained = True

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

st.set_page_config(page_title="Income Classifier", layout="centered")
st.title("💼 Income Classification App")

if trained:
    st.markdown(f"✅ **Model Loaded Successfully**")

def user_input():
    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", encoders["education"].classes_)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    hours = st.slider("Hours Worked Per Week", 1, 99, 40)
    gender = st.selectbox("Gender", encoders["gender"].classes_)

    input_dict = {
        "age": age,
        "education": encoders["education"].transform([education])[0],
        "marital-status": encoders["marital-status"].transform([marital_status])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "hours-per-week": hours,
        "gender": encoders["gender"].transform([gender])[0],
    }
    readable = {
        "Age": age,
        "Education": education,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Hours/Week": hours,
        "Gender": gender,
    }
    return pd.DataFrame([input_dict]), readable

input_df, readable_input = user_input()

# Ensure correct column structure and no NaNs
try:
    input_df = input_df[FEATURE_COLUMNS]

    if input_df.isnull().values.any():
        st.error("❌ Some required fields are missing. Please complete all inputs.")
    else:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        label = encoders["income"].inverse_transform([pred])[0]
        confidence = prob[pred] * 100

        st.success(f"💰 Predicted Income Category: `{label}`")
        st.info(f"🔐 Confidence: `{confidence:.2f}%`")

        st.markdown("### 🧾 Your Inputs")
        st.table(readable_input)

        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Category": encoders["income"].classes_,
            "Probability": prob
        })
        st.bar_chart(prob_df.set_index("Category"))
except KeyError as e:
    st.error(f"❌ Missing required column: {e}")
except Exception as e:
    st.error("🚨 An unexpected error occurred during prediction.")
    st.exception(e)
st.sidebar.header("🔍 Model & Encoders")
st.sidebar.write("### 📦 Model Summary")
# Safely display key parameters
st.sidebar.header("Model Overview")
st.sidebar.text("RandomForestClassifier:")
st.sidebar.markdown(f"- n_estimators: `{model.n_estimators}`")
st.sidebar.markdown(f"- max_depth: `{model.max_depth}`")
st.sidebar.markdown(f"- random_state: `{model.random_state}`")



st.sidebar.write("### 🔤 Encoders")
for col, le in encoders.items():
    st.sidebar.write(f"**{col}**: {list(le.classes_)}")
