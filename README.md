# 🚢 Titanic Survival Predictor — Streamlit ML Web App

An interactive web app that wraps a trained scikit-learn model with input widgets and a real-time prediction. 

**Live app:** https://customer-segmentation-app-gqigbpbpgelhvg3v7tekm3.streamlit.app/

## Features
- Streamlit interface with selectbox, radio and slider inputs (sensible min / max / defaults)
- Model saved and loaded with `joblib`
- `@st.cache_resource` so the model loads only once
- Prediction shown with `st.metric` and a progress bar, plus an educational caption
- Friendly error messages if the model file is missing or the input is invalid (no stack traces)
- No personal data collected

## Project structure
```
titanic-streamlit-app/
├── app.py            # Streamlit app
├── train.py          # Trains the model and saves model.joblib
├── model.joblib      # Saved model (Pipeline: imputer + RandomForest)
├── data/titanic.csv  # Public Titanic dataset (891 passengers)
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
python train.py          # optional: re-trains and re-creates model.joblib
streamlit run app.py
```

## How it works
1. `train.py` trains a Random Forest on `pclass`, `sex`, `age`, `fare` and saves it with `joblib.dump`.
2. `app.py` loads the model once (cached), builds a one-row DataFrame in the **same feature order** as training, calls `predict_proba`, and shows the survival probability.

Feature order (must be identical in training and the app): `["pclass", "sex", "age", "fare"]`. Sex is encoded as `female = 1`, `male = 0`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**, choose the repository, branch `main`, and main file `app.py`.
4. Deploy, then test the public link on your phone.

> If the deploy fails, pin package versions in `requirements.txt` (use the versions printed by `pip freeze`).

## Disclaimer
This is a learning exercise. It describes a historical dataset and is not advice.
