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
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

model = None
encoders = None
trained = False

# ----------------- Train Model ------------------
def train_model():
    df = pd.read_csv(CSV_PATH)

    X = df.drop("income", axis=1)
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

    # Compatibility-safe RMSE
    mse = mean_squared_error(y_test, clf.predict(X_test))
    rmse = mse ** 0.5

    return clf, enc, accuracy_score(y_test, clf.predict(X_test)), rmse

# ----------------- Load or Train (Safe) ------------------
def safe_load_or_train():
    global model, encoders, trained
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)

            # Check for version compatibility
            if hasattr(model, '_sklearn_version'):
                model_version = model._sklearn_version
                current_version = sklearn.__version__
                if model_version != current_version:
                    raise ValueError(f"Incompatible model version: trained on {model_version}, current is {current_version}")
            trained = True
        else:
            raise FileNotFoundError("Model files not found")
    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {e} - Re-training now...")
        model, encoders, acc, rmse = train_model()
        trained = True

# Call safe load/train
safe_load_or_train()

# ----------------- Streamlit UI ------------------
st.set_page_config(page_title="Income Classifier", layout="centered")
st.title("💼 Income Classification App")

if trained:
    st.markdown("✅ **Model Ready for Prediction**")

# ----------------- User Input ------------------
def user_input():
    age = st.slider("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
    fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1000000, value=300000)
    education = st.selectbox("Education", encoders["education"].classes_)
    education_num = st.slider("Education Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
    race = st.selectbox("Race", encoders["race"].classes_)
    gender = st.selectbox("Gender", encoders["sex"].classes_)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", encoders["native-country"].classes_)

    input_dict = {
        "age": age,
        "workclass": encoders["workclass"].transform([workclass])[0],
        "fnlwgt": fnlwgt,
        "education": encoders["education"].transform([education])[0],
        "educational-num": education_num,
        "marital-status": encoders["marital-status"].transform([marital_status])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "relationship": encoders["relationship"].transform([relationship])[0],
        "race": encoders["race"].transform([race])[0],
        "sex": encoders["sex"].transform([gender])[0],
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": encoders["native-country"].transform([native_country])[0],
    }

    readable_input = {
        "Age": age,
        "Workclass": workclass,
        "Fnlwgt": fnlwgt,
        "Education": education,
        "Education Num": education_num,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Relationship": relationship,
        "Race": race,
        "Gender": gender,
        "Capital Gain": capital_gain,
        "Capital Loss": capital_loss,
        "Hours per Week": hours_per_week,
        "Native Country": native_country
    }

    return pd.DataFrame([input_dict]), readable_input

# ----------------- Prediction ------------------
input_df, readable_input = user_input()

try:
    input_df = input_df[FEATURE_COLUMNS]

    if input_df.isnull().values.any():
        st.error("❌ Some required fields are missing.")
    else:
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
        except AttributeError:
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
for col, le in encoders.items():
    st.sidebar.markdown(f"**{col}**: {list(le.classes_)}")
