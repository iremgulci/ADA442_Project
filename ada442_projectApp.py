import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model (pipeline)
model = joblib.load("ada442_projectModel.pkl")

# Title
st.title("ADA442 Project - Bank Term Deposit Subscription Predictor")
st.write("Fill in the following client and economic data to predict the likelihood of subscription.")

# Load test data (you need to have this dataset available)
# For this example, we'll generate some dummy data similar to the input features.
# In practice, you would load the actual test dataset.
test_data = pd.read_csv("test_data.csv")  # Replace with your actual test dataset path

# Function to fill form with example values
def fill_example_data(index):
    example = test_data.iloc[index]
    st.session_state.age = example['age']
    st.session_state.job = example['job']
    st.session_state.marital = example['marital']
    st.session_state.education = example['education']
    st.session_state.default = example['default']
    st.session_state.housing = example['housing']
    st.session_state.loan = example['loan']
    st.session_state.contact = example['contact']
    st.session_state.month = example['month']
    st.session_state.day_of_week = example['day_of_week']
    st.session_state.duration = example['duration']
    st.session_state.campaign = example['campaign']
    st.session_state.pdays = example['pdays']
    st.session_state.previous = example['previous']
    st.session_state.poutcome = example['poutcome']
    st.session_state.emp_var_rate = example['emp.var.rate']
    st.session_state.cons_price_idx = example['cons.price.idx']
    st.session_state.cons_conf_idx = example['cons.conf.idx']
    st.session_state.euribor3m = example['euribor3m']
    st.session_state.nr_employed = example['nr.employed']
    return example

# Client data
age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age")
job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
], key="job")
marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"], key="marital")
education = st.selectbox("Education", [
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown"
], key="education")
default = st.selectbox("Default on Credit?", ["no", "yes", "unknown"], key="default")
housing = st.selectbox("Housing Loan?", ["no", "yes", "unknown"], key="housing")
loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"], key="loan")

# Contact info
contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"], key="contact")
month = st.selectbox("Last Contact Month", [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
], key="month")
day_of_week = st.selectbox("Last Contact Day of Week", ["mon", "tue", "wed", "thu", "fri"], key="day_of_week")
duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, value=0, key="duration")

# Campaign info
campaign = st.number_input("Number of Contacts During This Campaign", min_value=1, value=1, key="campaign")
pdays = st.number_input("Days Since Last Contact (999 = never contacted)", min_value=0, value=999, key="pdays")
previous = st.number_input("Previous Contacts", min_value=0, value=0, key="previous")
poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "nonexistent", "success"], key="poutcome")

# Economic indicators
emp_var_rate = st.number_input("Employment Variation Rate (e.g., -1.8)", value=1.1, key="emp_var_rate")
cons_price_idx = st.number_input("Consumer Price Index (e.g., 92.893)", value=93.2, key="cons_price_idx")
cons_conf_idx = st.number_input("Consumer Confidence Index (e.g., -46.2)", value=-36.4, key="cons_conf_idx")
euribor3m = st.number_input("Euribor 3-Month Rate (e.g., 0.634)", value=4.9, key="euribor3m")
nr_employed = st.number_input("Number of Employees (e.g., 5191)", value=5195, key="nr_employed")

# Example Buttons (1, 2, 3)
st.subheader("Select a Test Example to Fill:")
for i in range(3):
    if st.button(f"Example {i + 1}"):
        example = fill_example_data(i)
        st.write(f"Loaded Example {i + 1}:")
        st.write(example)

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
    probability = model.predict_proba(input_data)[0][1]

    # Display real outcome for the selected example
    real_outcome = example['y']  # Assuming the real outcome is in the column 'y'
    prediction_result = "SUBSCRIBE" if prediction == 1 else "NOT SUBSCRIBE"
    
    # Show result
    if prediction == 1:
        st.success(f"Prediction: The client is predicted to {prediction_result} to a term deposit. (Probability: {probability:.2%})")
    else:
        st.error(f"Prediction: The client is predicted NOT to subscribe. (Probability: {probability:.2%})")

    # Show real outcome
    if real_outcome == 1:
        st.success(f"Actual Outcome: The client actually subscribed.")
    else:
        st.error(f"Actual Outcome: The client did not subscribe.")
