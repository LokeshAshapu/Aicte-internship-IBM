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
DATA_PATH = "data.csv"  # Update this to the correct file path

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


safe_load_or_train()

# ----------------- Streamlit UI ------------------
st.set_page_config(page_title="Income Classifier", layout="centered")
st.title("💼 Income Classification App")

if trained:
    st.markdown("✅ **Model Ready for Prediction**")

def user_input():
    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", encoders["education"].classes_)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    hours = st.slider("Hours Worked Per Week", 1, 99, 40)
    gender = st.selectbox("Gender", encoders["gender"].classes_)

   input_dict = {
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "educational-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": gender,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}

input_df = pd.DataFrame([input_dict])

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

# ----------------- Prediction ------------------
try:
    input_df = input_df[FEATURE_COLUMNS]

    if input_df.isnull().values.any():
        st.error("❌ Some required fields are missing.")
    else:
        # Some older model attributes might break predict_proba in newer versions
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
for col, le in encoders.items():
    st.sidebar.markdown(f"**{col}**: {list(le.classes_)}")
