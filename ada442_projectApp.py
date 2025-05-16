import streamlit as st
import pandas as pd
import joblib
import numpy as np


def yes_no_to_binary(x):
    return x.applymap(lambda val: 1 if val == 'yes' else 0).astype(int)

def log_transform_func(df):
    return df.assign(
        campaign=np.log1p(df['campaign']),
        previous=np.log1p(df['previous'])
    )

# Load the trained model (pipeline)
model = joblib.load("best_model.pkl")

# Title
st.title("ADA442 Project - Bank Term Deposit Subscription Predictor")
st.write("Fill in the following client and economic data to predict the likelihood of subscription.")

# Client data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
])
marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
education = st.selectbox("Education", [
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown"
])
default = st.selectbox("Default on Credit?", ["no", "yes", "unknown"])
housing = st.selectbox("Housing Loan?", ["no", "yes", "unknown"])
loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"])

# Contact info
contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
month = st.selectbox("Last Contact Month", [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
])
day_of_week = st.selectbox("Last Contact Day of Week", ["mon", "tue", "wed", "thu", "fri"])
duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, value=0)

# Campaign info
campaign = st.number_input("Number of Contacts During This Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact (999 = never contacted)", min_value=0, value=999)
previous = st.number_input("Previous Contacts", min_value=0, value=0)
poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "nonexistent", "success"])

# Economic indicators
emp_var_rate = st.number_input("Employment Variation Rate (e.g., -1.8)", value=1.1)
cons_price_idx = st.number_input("Consumer Price Index (e.g., 92.893)", value=93.2)
cons_conf_idx = st.number_input("Consumer Confidence Index (e.g., -46.2)", value=-36.4)
euribor3m = st.number_input("Euribor 3-Month Rate (e.g., 0.634)", value=4.9)
nr_employed = st.number_input("Number of Employees (e.g., 5191)", value=5195)

# Predict button
if st.button("Predict"):
    # Construct input DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.write(f"The client is predicted to SUBSCRIBE to a term deposit.")
    else:
        st.write(f"The client is predicted NOT to subscribe.")
        
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.success(f"The client is predicted to SUBSCRIBE to a term deposit. (Probability: {probability:.2%})")
    else:
        st.error(f"The client is predicted NOT to subscribe. (Probability: {probability:.2%})")
    
st.header("Test Set Example Evaluation")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Choose 5 static example indices (could be random or the first 5)
example_indices = [15, 16, 17, 18, 19]

# Layout buttons in a row
cols = st.columns(len(example_indices))
for i, idx in enumerate(example_indices):
    with cols[i]:
        if st.button(f"Example {idx}"):
            # Predict
            test_sample = X_test.iloc[[idx]]
            pred = model.predict(test_sample)[0]
            pred_proba = model.predict_proba(test_sample)[0]  # probabilities for each class
            actual = y_test.iloc[idx][0] if isinstance(y_test.iloc[idx], pd.Series) else y_test.iloc[idx]

            # Show prediction result
            st.subheader(f"Result for Example {idx}")
            # Map numeric predictions to labels
            label_map = {0: "NOT SUBSCRIBE", 1: "SUBSCRIBE"}
            pred_label = label_map.get(pred, "Unknown")
            actual_label = label_map.get(actual, "Unknown")

            # Show result with probabilities
            if pred == actual:
                st.markdown(f"✅ **Correct Prediction:** {pred_label}")
            else:
                st.markdown(f"""
                    ❌ *Incorrect Prediction*  
                    **Predicted:** {pred_label}  
                    **Actual:** {actual_label}
                    """)

            st.markdown(f"**Prediction probabilities:**")
            st.write(f"NOT SUBSCRIBE: {pred_proba[0]:.3f}")
            st.write(f"SUBSCRIBE: {pred_proba[1]:.3f}")

