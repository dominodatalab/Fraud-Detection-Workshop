# Delivery And Hosting
In this phase, we will deploy the trained fraud detection model for production use. We will deploy the model as a REST API endpoint and create an interactive Streamlit dashboard for business user access.

## Exercise Instructions

### Step 1: Deploy Model Endpoints

The fraud detection application requires 4 separate endpoints:
1. Feature scaling/preprocessing pipeline
2. XGBoost classifier
3. AdaBoost classifier  
4. GaussianNB classifier

#### 1.1 Navigate to Model Registry
- Go to **Model Registry** in the left navigation panel
- Locate your registered models (created in Exercise 4):
  - `CC_Fraud_Feature_Scaling` (preprocessing pipeline)
  - `fraud_detection_classifier` (best classifier from training)
- Note the model names and version numbers

#### 1.2 Create Feature Scaling Endpoint
- Navigate to **Deployments > Endpoints**
- Click **Create Domino Endpoint**
- Configure the preprocessing endpoint:
  - **Name**: `fraud-feature-scaling`
  - **Description**: `Feature scaling and preprocessing for fraud detection`
  - **Model Source**: Select `CC_Fraud_Feature_Scaling` model
  - **Request Type**: Synchronous
  - **Compute Environment**: Select environment with scikit-learn dependencies

#### 1.3 Create Classifier Endpoints
Repeat endpoint creation for each classifier model:
- **XGBoost Endpoint**: `fraud-xgboost-classifier`
- **AdaBoost Endpoint**: `fraud-adaboost-classifier`  
- **GaussianNB Endpoint**: `fraud-gaussiannb-classifier`

#### 1.4 Model Input Schemas

**Feature Scaling Endpoint Input:**
```json
{
  "data": {
    "Time": "1699123456.789",
    "Amount": "150.75",
    "Age": "35",
    "Tenure": "2.5",
    "MerchantRisk": "0.2",
    "DeviceTrust": "1.1",
    "Txn24h": "3",
    "Avg30d": "245.50",
    "IPReputation": "0.8",
    "Latitude": "40.7128",
    "Longitude": "-74.0060",
    "DistFromHome": "0.5",
    "Hour": "14",
    "TxType": "purchase",
    "DeviceType": "mobile",
    "MerchantCat": "grocery",
    "Channel": "contactless",
    "CardPresent": "1",
    "amount_vs_avg30d_ratio": "0.61",
    "risk_score": "0.50",
    "trust_score": "0.90",
    "generation": "Millennial"
  }
}
```

**Classifier Endpoint Input:**
```json
{
  "data": {
    "num__Time": 0.123,
    "num__Amount": -0.456,
    "num__Age": 0.789,
    "num__Tenure": -0.234,
    "num__MerchantRisk": 0.567,
    "num__DeviceTrust": -0.890,
    "num__Txn24h": 0.345,
    "num__Avg30d": -0.678,
    "num__IPReputation": 0.901,
    "num__Latitude": -0.432,
    "num__Longitude": 0.765,
    "num__DistFromHome": -0.321,
    "num__Hour": 0.654,
    "num__CardPresent": 0.987,
    "num__amount_vs_avg30d_ratio": -0.210,
    "num__risk_score": 0.543,
    "num__trust_score": -0.876,
    "cat__TxType_payment": 0,
    "cat__TxType_purchase": 1,
    "cat__TxType_transfer": 0,
    "cat__TxType_withdrawal": 0,
    "cat__DeviceType_ATM": 0,
    "cat__DeviceType_POS": 0,
    "cat__DeviceType_desktop": 0,
    "cat__DeviceType_mobile": 1,
    "cat__DeviceType_web": 0,
    "cat__MerchantCat_clothing": 0,
    "cat__MerchantCat_electronics": 0,
    "cat__MerchantCat_entertainment": 0,
    "cat__MerchantCat_gas": 0,
    "cat__MerchantCat_grocery": 1,
    "cat__MerchantCat_restaurant": 0,
    "cat__MerchantCat_travel": 0,
    "cat__MerchantCat_utilities": 0,
    "cat__Channel_chip": 0,
    "cat__Channel_contactless": 1,
    "cat__Channel_in-store": 0,
    "cat__Channel_online": 0,
    "cat__generation_Baby Boomer": 0,
    "cat__generation_Generation X": 0,
    "cat__generation_Generation Z": 0,
    "cat__generation_Millennial": 1
  }
}
```

#### 1.5 Deploy and Test Endpoints
- Deploy each endpoint and wait for status to show "Running"
- Test the feature scaling endpoint with raw transaction data
- Test classifier endpoints with scaled feature data
- Copy all endpoint URLs (format: `https://se-demo.domino.tech:443/models/{model-id}/latest/model`)

#### 1.6 Configure Application Endpoints
Create `app_config.py` file based on the template:
```python
# Copy from app_config_template.py and update with your actual endpoint URLs
FEATURE_SCALING_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-feature-scaling-model-id}/latest/model"
FEATURE_SCALING_AUTH = "your-auth-token-here"

XGBOOST_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-xgboost-model-id}/latest/model"
XGBOOST_AUTH = "your-auth-token-here"

ADABOOST_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-adaboost-model-id}/latest/model"  
ADABOOST_AUTH = "your-auth-token-here"

GAUSSIANNB_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-gaussiannb-model-id}/latest/model"
GAUSSIANNB_AUTH = "your-auth-token-here"
```

### Step 2: Launch Streamlit Dashboard

#### 2.1 Development and Testing in Workspace

For development and testing, you can run the Streamlit app directly in your Domino Workspace:

**Generate the workspace app URL:**
```bash
echo -e "import os\nprint('https://your-domino-url/{}/{}/notebookSession/{}/proxy/8501/'.format(os.environ['DOMINO_PROJECT_OWNER'], os.environ['DOMINO_PROJECT_NAME'], os.environ['DOMINO_RUN_ID']))" | python3
```
*Note: Replace `your-domino-url` with your actual Domino domain (e.g., `company.domino.tech`)*

**Start the application:**
```bash
streamlit run app.py
```

**Access the app:**
- Copy the generated URL from the first command
- Open it in your browser to access the live app
- The URL format will be: `https://your-domino-url/{owner}/{project}/notebookSession/{run-id}/proxy/8501/`

#### 2.2 Access Dashboard
- Open the provided Streamlit URL from the command above
- The dashboard provides an interactive interface for testing fraud detection
- Test various transaction scenarios using the form inputs

#### 2.3 Publish App for Organization Access

To publish the Streamlit app for broader organizational access:

1. **Navigate to Apps**: Go to **Deploy > Apps** in your project
2. **Configure App**:
   - **Name**: `Fraud Detection Dashboard`
   - **Description**: `Interactive fraud detection application`
   - **Launch File**: Select `app.sh` (already created)
3. **Runtime Settings**:
   - **Compute Environment**: Select environment with Streamlit dependencies
   - **Hardware Tier**: Choose appropriate tier for expected usage
4. **Access Control**: Set visibility (Domino users, restricted, etc.)
5. **Publish**: Click publish to deploy the app

The app will be available at a dedicated URL for organizational access.

### Step 3: Verify Deployment

#### 3.1 Endpoint Health Check
- Verify endpoint status in Deployments dashboard
- Check endpoint logs for any errors
- Validate response format matches expected output

#### 3.2 Integration Testing
- Test endpoint with various transaction amounts and risk levels
- Verify fraud predictions align with expected model behavior
- Document any performance observations

This concludes the "Delivery & Hosting" section of the workshop.

---

## New Domino Concepts

**Model Endpoints:**
> Model Endpoints are REST API services that automatically deploy trained models as scalable, production-ready APIs with built-in load balancing, monitoring, and versioning capabilities. This enables data scientists to instantly serve predictions to applications and systems without writing deployment code or managing infrastructure, while IT maintains governance and security controls.

**Hosted Applications:**
> Hosted Applications allow users to deploy and share interactive web applications (built with frameworks like Streamlit, Dash, or Flask) directly from their Domino projects with automatic scaling and authentication. This empowers data scientists to create self-service analytics tools and model interfaces for business users without requiring web development expertise or separate hosting infrastructure.

 
