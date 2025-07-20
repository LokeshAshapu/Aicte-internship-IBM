import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# ----------------- Constants ------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "income_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
CSV_PATH = "adult 3.csv"
FEATURE_COLUMNS = [
    'age', 'education', 'occupation', 'hours-per-week', 'gender'
]

model = None
encoders = None
trained = False

# ----------------- Train Model ------------------
def train_model():
    df = pd.read_csv(CSV_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["income"]

    enc = {}
    for col in X.select_dtypes(include=["object"]).columns:
        enc[col] = LabelEncoder()
        X[col] = enc[col].fit_transform(X[col])
    enc["income"] = LabelEncoder()
    y = enc["income"].fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(enc, ENCODER_PATH)

    mse = mean_squared_error(y_test, clf.predict(X_test))
    rmse = mse ** 0.5

    return clf, enc, accuracy_score(y_test, clf.predict(X_test)), rmse

# ----------------- Load or Train (Safe) ------------------
def safe_load_or_train():
    global model, encoders, trained
    try:
        retrain = False

        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)

            if hasattr(model, '_sklearn_version'):
                model_version = model._sklearn_version
                current_version = sklearn.__version__
                if model_version != current_version:
                    st.warning(f"Incompatible model version: trained on {model_version}, current is {current_version}")
                    retrain = True

            for col in ['education', 'occupation', 'gender', 'income']:
                if col not in encoders:
                    st.warning(f"Encoder for '{col}' is missing. Triggering retrain.")
                    retrain = True
                    break

        else:
            retrain = True

        if retrain:
            model, encoders, acc, rmse = train_model()
            trained = True
        else:
            trained = True

    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {e} - Re-training now...")
        model, encoders, acc, rmse = train_model()
        trained = True

safe_load_or_train()

# ----------------- Streamlit UI ------------------
st.set_page_config(page_title="Income Classifier", layout="centered")
st.title("💼 Income Classification App")

if trained:
    st.markdown("✅ **Model Ready for Prediction**")

def user_input():
    st.markdown("### 📝 Enter your details below")

    if not encoders:
        st.error("❌ Encoders not available. Please retrain the model.")
        return None, None

    required_keys = ['education', 'occupation', 'gender']
    missing = [key for key in required_keys if key not in encoders]
    if missing:
        st.error(f"❌ Missing encoders for: {', '.join(missing)}. Please check your dataset or retrain the model.")
        return None, None

    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", encoders["education"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    gender = st.selectbox("Gender", encoders["gender"].classes_)

    input_dict = {
        "age": age,
        "education": encoders["education"].transform([education])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "hours-per-week": hours_per_week,
        "gender": encoders["gender"].transform([gender])[0],
    }

    readable_input = {
        "Age": age,
        "Education": education,
        "Occupation": occupation,
        "Hours per Week": hours_per_week,
        "Gender": gender
    }

    return pd.DataFrame([input_dict]), readable_input

# ----------------- Prediction ------------------
input_df, readable_input = user_input()

if input_df is not None:
    try:
        input_df = input_df[FEATURE_COLUMNS]

        if input_df.isnull().values.any():
            st.error("❌ Some required fields are missing.")
        else:
            try:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
            except AttributeError as err:
                st.error("🚨 Prediction failed due to model incompatibility. Retraining...")
                model, encoders, acc, rmse = train_model()
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]

            label = encoders["income"].inverse_transform([pred])[0]
            confidence = prob[pred] * 100

            st.success(f"💰 Predicted Income Category: `{label}`")
            st.info(f"🔐 Confidence: `{confidence:.2f}%`")

            st.markdown("### 🧾 Your Inputs")
            st.table(pd.DataFrame([readable_input]))

            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Category": encoders["income"].classes_,
                "Probability": prob
            })
            st.bar_chart(prob_df.set_index("Category"))

    except Exception as e:
        st.error("🚨 Unexpected error during prediction.")
        st.exception(e)

# ----------------- Sidebar ------------------
st.sidebar.header("🔍 Model & Encoders")
st.sidebar.subheader("Model Parameters")

if model:
    st.sidebar.markdown(f"- n_estimators: `{getattr(model, 'n_estimators', 'N/A')}`")
    st.sidebar.markdown(f"- max_depth: `{getattr(model, 'max_depth', 'N/A')}`")

st.sidebar.subheader("Label Encoders")

if encoders:
    for col, le in encoders.items():
        st.sidebar.markdown(f"**{col}**: {list(le.classes_)}")
else:
    st.sidebar.warning("Encoders not loaded. Please retrain the model.")

if st.sidebar.button("🔁 Retrain Model"):
    with st.spinner("Retraining model..."):
        model, encoders, acc, rmse = train_model()
        st.sidebar.success("✅ Model retrained!")
