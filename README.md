# Super-OTP-Predictor
The objective of this project is to simulate and build a machine learning-based decision system that predicts whether a user is eligible for Super OTP (1) or should default to the traditional Standard OTP (0).# 🔐 Super OTP Eligibility Predictor

A machine learning-based risk engine that predicts whether a user is eligible for **Super OTP** — an instant in-app OTP generated based on behavior and context — or should fall back to the traditional **Standard OTP**.

---

## 📌 Problem Statement

Traditional OTP systems are slow and prone to interception (e.g., SIM swap). Modern apps aim to reduce friction by generating **Super OTPs** internally when user behavior and transaction context seem safe.

This project simulates user behavior, trains a binary classification model, and deploys an interactive app to predict OTP eligibility in real time.

---

## 🎯 Objective

- Simulate a realistic dataset of 10,000 transaction entries.
- Train a binary classification model to predict `eligible_for_super_otp` (1 or 0).
- Build a UI using Streamlit for real-time prediction and testing.

---

## 🧠 Eligibility Logic

A user is eligible for **Super OTP** (`1`) if all of the following are true:
- Device is **Mobile**
- GPS matches billing address
- Logged in ≥ **5 minutes** ago
- App usage today ≥ **20 minutes**
- Battery ≥ **20%**
- No past fraud history
- Network is **WiFi** or **4G**
- *Payment amount ≤ ₹5000* 

Otherwise, Standard OTP (`0`) is triggered.

---

## 📊 Dataset Features

| Feature               | Type        | Description                                  |
|-----------------------|-------------|----------------------------------------------|
| user_id               | String      | Anonymized user ID                           |
| device_type           | Categorical | Mobile / Desktop                             |
| location_match        | Binary      | 1 if GPS matches billing address             |
| app_login_duration    | Numeric     | Minutes since login                          |
| app_usage_today       | Numeric     | Minutes of app usage today                   |
| payment_amount        | Numeric     | Transaction amount in INR                    |
| transaction_hour      | Numeric     | Hour of transaction (0–23)                   |
| past_fraud_flag       | Binary      | 1 if previous fraud detected                 |
| network_type          | Categorical | WiFi / 4G / None                             |
| os_version            | Categorical | e.g., Android 12, iOS 16                     |
| battery_level         | Numeric     | 0–100% battery level                         |
| eligible_for_super_otp| Binary      | Target variable (1 = Super OTP, 0 = Standard)|

---

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- Faker (for simulating data)
- Streamlit (for interactive app)
- Matplotlib / Seaborn (for analysis)

---

## 🚀 Streamlit App

A real-time prediction interface using `streamlit_app.py`.

### ▶️ To Run Locally:
```bash
# Create environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

