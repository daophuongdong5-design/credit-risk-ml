import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_FILE = "credit_risk_model.pkl"

# Train model if not exist
if not os.path.exists(MODEL_FILE):

    df = pd.read_csv("credit_risk_dataset_10000.csv")

    le = LabelEncoder()
    df["loan_purpose"] = le.fit_transform(df["loan_purpose"])

    X = df.drop("default_risk", axis=1)
    y = df["default_risk"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    pickle.dump(model, open(MODEL_FILE, "wb"))

# Load model
model = pickle.load(open(MODEL_FILE,"rb"))
