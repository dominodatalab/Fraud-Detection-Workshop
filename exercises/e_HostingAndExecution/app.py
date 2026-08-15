import os
import sys
import streamlit as st
import time
import random
import pandas as pd
import numpy as np
import requests

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exercises.c_DataEngineering.data_engineering import add_derived_features

# Import configuration (fallback to environment variables if config file not available)
try:
    from app_config import (
        FEATURE_SCALING_ENDPOINT, FEATURE_SCALING_AUTH,
        MODEL_CONFIG
    )
    feature_scaling_endpoint = FEATURE_SCALING_ENDPOINT
    feature_scaling_auth = FEATURE_SCALING_AUTH
    model_scaling_dict = MODEL_CONFIG
    print("✅ Loaded configuration from app_config.py")
except ImportError:
    # Fallback to environment variables
    print("⚠️ app_config.py not found, using environment variables")
    feature_scaling_endpoint = os.environ.get('feature_scaling_endpoint', 'ENDPOINT_NOT_CONFIGURED')
    feature_scaling_auth = os.environ.get('feature_scaling_auth', 'AUTH_NOT_CONFIGURED')
    
    xgboost_endpoint = os.environ.get('xgboost_endpoint', 'ENDPOINT_NOT_CONFIGURED')
    xgboost_auth = os.environ.get('xgboost_auth', 'AUTH_NOT_CONFIGURED')
    
    adaboost_endpoint = os.environ.get('adaboost_endpoint', 'ENDPOINT_NOT_CONFIGURED')
    adaboost_auth = os.environ.get('adaboost_auth', 'AUTH_NOT_CONFIGURED')
    
    gaussiannb_endpoint = os.environ.get('gaussiannb_endpoint', 'ENDPOINT_NOT_CONFIGURED')
    gaussiannb_auth = os.environ.get('gaussiannb_auth', 'AUTH_NOT_CONFIGURED')
    
    model_scaling_dict = {
        'XG Boost': {
            'endpoint': xgboost_endpoint,
            'auth': xgboost_auth,
        },
        'ADA Boost': {
            'endpoint': adaboost_endpoint,
            'auth': adaboost_auth,
        },
        'GaussianNB': {
            'endpoint': gaussiannb_endpoint,
            'auth': gaussiannb_auth,
        }
    }

# Define schema once at module level
CLASSIFIER_SCHEMA = [
    'num__Time', 'num__Amount', 'num__Age', 'num__Tenure', 'num__MerchantRisk',
    'num__DeviceTrust', 'num__Txn24h', 'num__Avg30d', 'num__IPReputation',
    'num__Latitude', 'num__Longitude', 'num__DistFromHome', 'num__Hour',
    'num__CardPresent', 'num__amount_vs_avg30d_ratio', 'num__risk_score',
    'num__trust_score', 'cat__TxType_payment', 'cat__TxType_purchase',
    'cat__TxType_transfer', 'cat__TxType_withdrawal', 'cat__DeviceType_ATM',
    'cat__DeviceType_POS', 'cat__DeviceType_desktop', 'cat__DeviceType_mobile',
    'cat__DeviceType_web', 'cat__MerchantCat_clothing', 'cat__MerchantCat_electronics',
    'cat__MerchantCat_entertainment', 'cat__MerchantCat_gas', 'cat__MerchantCat_grocery',
    'cat__MerchantCat_restaurant', 'cat__MerchantCat_travel', 'cat__MerchantCat_utilities',
    'cat__Channel_chip', 'cat__Channel_contactless', 'cat__Channel_in-store',
    'cat__Channel_online', 'cat__generation_Baby Boomer', 'cat__generation_Generation X',
    'cat__generation_Generation Z', 'cat__generation_Millennial'
]


def scaled_data_to_classifier_format(scaled_data):
    """Convert scaled data array to classifier input format"""
    values = scaled_data[0]  # First (and only) row
    
    # Dynamically create the dictionary using zip
    classifier_data = dict(zip(CLASSIFIER_SCHEMA, values))
    return classifier_data
    

def create_transaction_data(amount, hour, tx_type, card_present, age, tenure, 
                          txn_24h, avg_30d, merchant_risk, device_trust, 
                          ip_reputation, dist_from_home, latitude, longitude, 
                          device_type, merchant_cat, channel):
    """Create a single-row DataFrame with transaction data"""
    
    # Create timestamp for current time (you can modify this as needed)
    current_time = time.time()
    
    # Create raw transaction data matching your expected structure
    raw_data = {
        'Time': current_time,
        'Amount': amount,
        'Age': age,
        'Tenure': tenure,
        'MerchantRisk': merchant_risk,
        'DeviceTrust': device_trust,
        'Txn24h': txn_24h,
        'Avg30d': avg_30d,
        'IPReputation': ip_reputation,
        'Latitude': latitude,
        'Longitude': longitude,
        'DistFromHome': dist_from_home,
        'Hour': hour,
        'TxType': tx_type,
        'DeviceType': device_type,
        'MerchantCat': merchant_cat,
        'Channel': channel,
        'CardPresent': card_present
    }
    
    # Create DataFrame
    df = pd.DataFrame([raw_data])
    
    # Add derived features
    df_with_features = add_derived_features(df)
    
    return df_with_features



st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        border: 2px solid #ff0000;
    }
    .safe-alert {
        background-color: #00cc88;
        color: white;
        border: 2px solid #00aa66;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔒 Fraud Detection App</h1>', unsafe_allow_html=True)

# Add model selection dropdown
st.subheader("🤖 Model Selection")
selected_model = st.selectbox(
    "Choose Classifier Model",
    options=list(model_scaling_dict.keys()),
    index=2  # Default to GaussianNB
)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Transaction Details")
    amount = st.number_input("Amount", min_value=0.01, value=50.0, step=0.01)
    hour = st.selectbox("Hour", range(24), index=12)
    tx_type = st.selectbox("Transaction Type", ["purchase", "transfer", "payment", "withdrawal"])
    card_present = st.selectbox("Card Present", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    st.subheader("👤 Customer Profile")
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure (years)", 0, 30, 5)
    txn_24h = st.number_input("Transactions in 24h", min_value=0, value=1, step=1)
    avg_30d = st.number_input("Avg 30d Amount", value=0.0, step=0.01)

with col3:
    st.subheader("🔍 Risk Factors")
    merchant_risk = st.slider("Merchant Risk", -3.0, 3.0, 0.0, 0.1)
    device_trust = st.slider("Device Trust", -3.0, 3.0, 0.0, 0.1)
    ip_reputation = st.slider("IP Reputation", -3.0, 3.0, 0.0, 0.1)
    dist_from_home = st.slider("Distance from Home", -2.0, 5.0, 0.0, 0.1)

# Location and merchant details
st.subheader("📍 Location & Merchant")
col4, col5 = st.columns(2)

with col4:
    latitude = st.number_input("Latitude", value=40.0, step=0.01)
    longitude = st.number_input("Longitude", value=-75.0, step=0.01)
    
with col5:
    device_type = st.selectbox("Device Type", ["mobile", "desktop", "ATM", "tablet"])
    merchant_cat = st.selectbox("Merchant Category", 
                               ["grocery", "gas", "electronics", "travel", "clothing", "entertainment", "restaurant"])
    channel = st.selectbox("Channel", ["online", "in-store", "contactless", "chip"])

# Prediction button and results
st.markdown("---")
predict_button = st.button("🔍 Predict Fraud Risk", type="primary")

if predict_button:
    # Show loading spinner
    with st.spinner("Analyzing transaction... 🔍"):
        time.sleep(2)  # Simulate API call

        
        # Create the transaction data with derived features
        transaction_df = create_transaction_data(
            amount=amount,
            hour=hour,
            tx_type=tx_type,
            card_present=card_present,
            age=age,
            tenure=tenure,
            txn_24h=txn_24h,
            avg_30d=avg_30d,
            merchant_risk=merchant_risk,
            device_trust=device_trust,
            ip_reputation=ip_reputation,
            dist_from_home=dist_from_home,
            latitude=latitude,
            longitude=longitude,
            device_type=device_type,
            merchant_cat=merchant_cat,
            channel=channel
        )
        
        print("Here's the data we have right now:")
        print(transaction_df.to_dict('records')[0])
        
        # Create JSON structure matching your expected format
        transaction_json = {
            "data": {
                "Time": str(transaction_df['Time'].iloc[0]),
                "Amount": str(transaction_df['Amount'].iloc[0]),
                "Age": str(transaction_df['Age'].iloc[0]),
                "Tenure": str(transaction_df['Tenure'].iloc[0]),
                "MerchantRisk": str(transaction_df['MerchantRisk'].iloc[0]),
                "DeviceTrust": str(transaction_df['DeviceTrust'].iloc[0]),
                "Txn24h": str(transaction_df['Txn24h'].iloc[0]),
                "Avg30d": str(transaction_df['Avg30d'].iloc[0]),
                "IPReputation": str(transaction_df['IPReputation'].iloc[0]),
                "Latitude": str(transaction_df['Latitude'].iloc[0]),
                "Longitude": str(transaction_df['Longitude'].iloc[0]),
                "DistFromHome": str(transaction_df['DistFromHome'].iloc[0]),
                "Hour": str(transaction_df['Hour'].iloc[0]),
                "TxType": transaction_df['TxType'].iloc[0],
                "DeviceType": transaction_df['DeviceType'].iloc[0],
                "MerchantCat": transaction_df['MerchantCat'].iloc[0],
                "Channel": transaction_df['Channel'].iloc[0],
                "CardPresent": str(transaction_df['CardPresent'].iloc[0]),
                "amount_vs_avg30d_ratio": str(round(transaction_df['amount_vs_avg30d_ratio'].iloc[0], 2)),
                "risk_score": str(round(transaction_df['risk_score'].iloc[0], 2)),
                "trust_score": str(round(transaction_df['trust_score'].iloc[0], 2)),
                "generation": transaction_df['generation'].iloc[0]
            }
        }

        scaled_transaction = None
        fraud_prediction = None

        # Make API call for input scaling
        try:
            response = requests.post(
                feature_scaling_endpoint,
                auth=(
                    feature_scaling_auth,
                    feature_scaling_auth
                ),
                json=transaction_json
            )
            
            if response.status_code == 200:
                resp = response.json()
                scaled_transaction = resp['result']
                print('scaled_transaction = ')
                print(scaled_transaction)

                # Convert scaled data to classifier format
                classifier_input = scaled_data_to_classifier_format(scaled_transaction)
                print('classifier_input = ')
                print(classifier_input)

                # Get selected model endpoint and auth
                selected_endpoint = model_scaling_dict[selected_model]['endpoint']
                selected_auth = model_scaling_dict[selected_model]['auth']

                # Make API call to selected classifier
                try:
                    classifier_response = requests.post(
                        selected_endpoint,
                        auth=(selected_auth, selected_auth),
                        json={"data": classifier_input}
                    )
                    
                    if classifier_response.status_code == 200:
                        classifier_resp = classifier_response.json()
                        print('resp')
                        print(classifier_resp)
                        fraud_prediction = classifier_resp['result']
                        print('fraud_prediction = ')
                        print(fraud_prediction)
                    else:
                        st.error(f"Classifier API Error: {classifier_response.status_code}")
                        print(f"Classifier error: {classifier_response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Classifier Connection Error: {str(e)}")


                
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")

        
        
        # Dummy prediction logic (you can replace this with actual model)
        final_risk_score = random.uniform(0, 1)
        
        # Simple heuristic for demo
        risk_factors = [
            amount > 500,
            merchant_risk > 1.0,
            device_trust < -1.0,
            ip_reputation < -1.0,
            dist_from_home > 2.0,
            hour in [0, 1, 2, 3, 4, 5, 23]
        ]
        
        final_risk_score = sum(risk_factors) / len(risk_factors)
        is_fraud = final_risk_score > 0.4
        
        # Display results
        if is_fraud:
            st.markdown(f"""
            <div class="prediction-box fraud-alert">
                <h2>⚠️ FRAUD ALERT</h2>
                <h3>Risk Score: {final_risk_score:.2%}</h3>
                <p>This transaction has been flagged as potentially fraudulent.</p>
                <p>Please review manually before processing.</p>
                <p><strong>Model Used:</strong> {selected_model}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box safe-alert">
                <h2>✅ TRANSACTION APPROVED</h2>
                <h3>Risk Score: {final_risk_score:.2%}</h3>
                <p>This transaction appears to be legitimate.</p>
                <p>Safe to process.</p>
                <p><strong>Model Used:</strong> {selected_model}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show risk factors breakdown
        st.subheader("📊 Risk Analysis")
        
        risk_data = {
            "Factor": ["High Amount", "Merchant Risk", "Device Trust", "IP Reputation", "Distance", "Unusual Hour"],
            "Score": [amount/1000, merchant_risk, device_trust, ip_reputation, dist_from_home, 1 if hour in [0,1,2,3,4,5,23] else 0],
            "Status": ["⚠️" if factor else "✅" for factor in risk_factors]
        }
        
        df = pd.DataFrame(risk_data)
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🔒 Fraud Detection System v1.0 | Built with Streamlit</p>",
    unsafe_allow_html=True
)