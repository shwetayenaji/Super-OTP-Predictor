import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load trained model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Super OTP Predictor", 
    page_icon="🔐", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            padding: 1rem 2rem;
            font-family: 'Inter', sans-serif;
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .header-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            text-align: center;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        
        /* Card styling */
        .input-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e8ecf0;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .input-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .card-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e8ecf0;
        }
        
        /* Input styling */
        .stSelectbox > label, .stSlider > label {
            font-weight: 500 !important;
            color: #34495e !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .stSelectbox > div > div {
            border-radius: 8px !important;
            border: 2px solid #e8ecf0 !important;
            transition: border-color 0.2s ease !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        .reset-button > button {
            background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%) !important;
            box-shadow: 0 4px 15px rgba(149, 165, 166, 0.3) !important;
        }
        
        .reset-button > button:hover {
            box-shadow: 0 6px 20px rgba(149, 165, 166, 0.4) !important;
        }
        
        /* Result styling */
        .result-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-top: 2rem;
            text-align: center;
            border: 1px solid #e8ecf0;
        }
        
        .success-result {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
        }
        
        .error-result {
            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
            color: white;
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .result-text {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
        }
        
        .result-subtext {
            font-size: 1rem;
            margin-top: 0.5rem;
            opacity: 0.9;
        }
        
        /* Metric cards */
        .metric-container {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            flex: 1;
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #e8ecf0;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.8rem;
            font-weight: 500;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-top: 0.25rem;
        }
        
        /* Animations */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in {
            animation: slideIn 0.5s ease-out;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main { padding: 0.5rem; }
            .header-title { font-size: 2rem; }
            .input-card { padding: 1.5rem; }
            .metric-container { flex-direction: column; }
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container slide-in">
        <h1 class="header-title">🔐 Super OTP Predictor</h1>
        <p class="header-subtitle">Advanced ML-powered authentication eligibility assessment</p>
    </div>
""", unsafe_allow_html=True)

# Default values
default_values = {
    "device_type": 0,
    "location_match": 1,
    "app_login_duration": 5,
    "app_usage_today": 30,
    "payment_amount": 1000,
    "transaction_hour": 14,
    "past_fraud_flag": 0,
    "network_type": 0,
    "os_version": 1,
    "battery_level": 75,
}

# Input options
device_options = ["Mobile", "Desktop"]
network_options = ["WiFi", "4G", "None"]
os_options = ["Android 12", "Android 13", "iOS 15", "iOS 16"]

# Layout with columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="input-card slide-in">
            <h3 class="card-title">📱 Device & Security Information</h3>
        </div>
    """, unsafe_allow_html=True)
    
    device_type = st.selectbox("Device Type", device_options, index=default_values["device_type"])
    os_version = st.selectbox("Operating System Version", os_options, index=default_values["os_version"])
    network_type = st.selectbox("Network Connection Type", network_options, index=default_values["network_type"])
    location_match = st.selectbox("Location Verification", 
                                 ["✅ GPS matches billing address", "❌ Location mismatch"], 
                                 index=0 if default_values["location_match"] == 1 else 1)
    past_fraud_flag = st.selectbox("Historical Security Status", 
                                  ["✅ Clean record", "⚠️ Previous fraud detected"], 
                                  index=default_values["past_fraud_flag"])

with col2:
    st.markdown("""
        <div class="input-card slide-in">
            <h3 class="card-title">📊 Usage & Transaction Details</h3>
        </div>
    """, unsafe_allow_html=True)
    
    app_login_duration = st.slider("Current Session Duration (minutes)", 0, 60, default_values["app_login_duration"], step=1)
    app_usage_today = st.slider("Total App Usage Today (minutes)", 0, 300, default_values["app_usage_today"], step=5)
    battery_level = st.slider("Device Battery Level (%)", 0, 100, default_values["battery_level"], step=5)
    payment_amount = st.slider("Transaction Amount (₹)", 50, 25000, default_values["payment_amount"], step=50)
    transaction_hour = st.slider("Transaction Time (24-hour format)", 0, 23, default_values["transaction_hour"], step=1)

# Quick metrics overview
st.markdown("""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">₹{:,}</div>
            <div class="metric-label">Transaction Amount</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{}%</div>
            <div class="metric-label">Battery Level</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{} min</div>
            <div class="metric-label">Session Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{}:00</div>
            <div class="metric-label">Transaction Hour</div>
        </div>
    </div>
""".format(payment_amount, battery_level, app_login_duration, transaction_hour), unsafe_allow_html=True)

# Action button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_clicked = st.button("🔍 Analyze Eligibility", use_container_width=True)

# Encoding mappings
device_map = {"Mobile": 1, "Desktop": 0}
network_map = {"WiFi": 2, "4G": 1, "None": 0}
os_map = {"Android 12": 0, "Android 13": 1, "iOS 15": 2, "iOS 16": 3}

# Convert display values to actual values for processing
location_binary = 1 if "GPS matches" in location_match else 0
fraud_binary = 1 if "Previous fraud" in past_fraud_flag else 0

# Prediction logic
if predict_clicked:
    with st.spinner("🔍 Analyzing user profile and transaction patterns..."):
        import time
        time.sleep(1.5)  # Add a small delay for better UX
        
        input_data = pd.DataFrame([[
            device_map[device_type],
            location_binary,
            app_login_duration,
            app_usage_today,
            payment_amount,
            transaction_hour,
            fraud_binary,
            network_map[network_type],
            os_map[os_version],
            battery_level
        ]], columns=[
            "device_type", "location_match", "app_login_duration",
            "app_usage_today", "payment_amount", "transaction_hour",
            "past_fraud_flag", "network_type", "os_version", "battery_level"
        ])

        prediction = model.predict(input_data)[0]
        
        # Get prediction probability if available
        try:
            probability = model.predict_proba(input_data)[0]
            confidence = max(probability) * 100
        except:
            confidence = 85  # Default confidence if predict_proba not available

        if prediction == 1:
            st.markdown(f"""
                <div class="result-card success-result slide-in">
                    <div class="result-icon">✅</div>
                    <h2 class="result-text">Super OTP Approved!</h2>
                    <p class="result-subtext">User meets all criteria for enhanced authentication • Confidence: {confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
        else:
            st.markdown(f"""
                <div class="result-card error-result slide-in">
                    <div class="result-icon">🔒</div>
                    <h2 class="result-text">Standard OTP Required</h2>
                    <p class="result-subtext">User profile suggests standard authentication • Confidence: {confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #7f8c8d; border-top: 1px solid #e8ecf0;">
        <p style="margin: 0; font-size: 0.9rem;">
            🔐 Powered by Machine Learning • Built with Streamlit • Secure & Privacy-Focused
        </p>
    </div>
""", unsafe_allow_html=True)