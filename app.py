"""Streamlit app: interactive Titanic survival predictor."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Must match the column order used in train.py
FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


@st.cache_resource
def load_model():
    """Load the model once and reuse it across reruns/clicks."""
    return joblib.load(MODEL_PATH)


st.title("🚢 Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list.")

# ---------- Inputs ----------
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2, help="1 = First, 3 = Third")
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col2:
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    fare = st.slider("Fare paid (£)", min_value=0, max_value=500, value=30)

# ---------- Prediction ----------
if st.button("Predict", type="primary"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file is missing. Run `python train.py` to train and save it first.")
        st.stop()
    except Exception:
        st.error("The model could not be loaded. Please retrain it with `python train.py`.")
        st.stop()

    try:
        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=FEATURES,
        )
        chance = float(model.predict_proba(row)[0][1])
    except Exception:
        st.warning("Those inputs could not be used. Please adjust them and try again.")
        st.stop()

    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.progress(chance)
    if chance >= 0.5:
        st.success("The model leans towards survival for this passenger.")
    else:
        st.info("The model leans towards non-survival for this passenger.")

    st.caption("A learning exercise. It describes a historical dataset and is not advice.")

with st.expander("About this model"):
    st.write(
        "A Random Forest classifier trained on the public Titanic passenger dataset "
        "(891 passengers) using ticket class, sex, age and fare. It is an educational "
        "example of deploying a machine learning model with Streamlit. No personal data "
        "is collected or stored."
    )
