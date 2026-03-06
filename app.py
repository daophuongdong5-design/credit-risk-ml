import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("credit_risk_model.pkl","rb"))

st.title("AI Credit Risk Scoring System")

st.write("Enter customer information")

income = st.number_input("Income",1000,20000)
employment = st.number_input("Employment Length (years)",0,40)
debt_ratio = st.slider("Debt Ratio",0.0,1.0)
credit_score = st.number_input("Credit Score",300,850)

loan_purpose = st.selectbox(
"Loan Purpose",
["home","car","education","business","personal"]
)

purpose_map = {
"home":0,
"car":1,
"education":2,
"business":3,
"personal":4
}

if st.button("Predict Credit Risk"):

    data = np.array([[
        income,
        employment,
        debt_ratio,
        credit_score,
        purpose_map[loan_purpose]
    ]])

    risk = model.predict_proba(data)[0][1]

    st.write("Risk Score:", round(risk*100,2),"%")

    if risk < 0.3:
        st.success("Loan Approved")

    elif risk < 0.6:
        st.warning("Need Manual Review")

    else:
        st.error("High Risk - Reject")
