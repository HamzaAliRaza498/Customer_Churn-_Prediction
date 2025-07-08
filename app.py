import streamlit as st
import pandas as pd
import joblib

# Load model and pipeline
model = joblib.load("customer_churn_model.pkl")
pipeline = joblib.load("churn_preprocessing_pipeline.pkl")

# Page title
st.title("Customer Churn Prediction App (Hamza Ali Raza)")
st.write("Upload a CSV file **or** manually enter customer data to predict churn.")

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(user_data)

        # Predict
        prepared_data = pipeline.transform(user_data)
        predictions = model.predict(prepared_data)
        proba = model.predict_proba(prepared_data)

        results = pd.DataFrame({
            "Churn Prediction": ["Yes" if p == 1 else "No" for p in predictions],
            "Churn Probability (%)": (proba[:, 1] * 100).round(2)
        })
        st.subheader("✅ Prediction Results")
        st.write(results)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ------------------ Manual Input ------------------
st.markdown("---")
st.subheader("✍️ Or Enter Customer Info Manually")

with st.form("manual_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiline = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    submit = st.form_submit_button("Predict Manually")

if submit:
    manual_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Predict
    prepared = pipeline.transform(manual_data)
    pred = model.predict(prepared)[0]
    prob = model.predict_proba(prepared)[0][1] * 100

    st.subheader("✅ Manual Prediction Result")
    st.write(f"**Churn Prediction:** {'Yes' if pred == 1 else 'No'}")
    st.write(f"**Churn Probability:** {prob:.2f}%")
