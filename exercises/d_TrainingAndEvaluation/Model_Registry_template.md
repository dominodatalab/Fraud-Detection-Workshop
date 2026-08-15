# Model Registry Template

## Model Information

**Model Name**: fraud_detection_classifier_v1  
**Model Type**: Binary Classification  
**Use Case**: Credit Card Fraud Detection  
**Training Date**: [AUTO-GENERATED]  
**Model Framework**: [XGBoost/AdaBoost/GaussianNB]  

## Model Specifications

### 1. Input Schema
```json
{
  "type": "object",
  "properties": {
    "Time": {"type": "number", "description": "Transaction timestamp"},
    "Amount": {"type": "number", "description": "Transaction amount"},
    "Age": {"type": "integer", "description": "Customer age"},
    "Tenure": {"type": "number", "description": "Customer tenure in years"},
    "MerchantRisk": {"type": "number", "description": "Merchant risk score"},
    "DeviceTrust": {"type": "number", "description": "Device trust score"},
    "Txn24h": {"type": "integer", "description": "Transactions in last 24 hours"},
    "Avg30d": {"type": "number", "description": "Average transaction amount in 30 days"},
    "IPReputation": {"type": "number", "description": "IP reputation score"},
    "Latitude": {"type": "number", "description": "Transaction latitude"},
    "Longitude": {"type": "number", "description": "Transaction longitude"},
    "DistFromHome": {"type": "number", "description": "Distance from home location"},
    "Hour": {"type": "integer", "description": "Hour of transaction"},
    "TxType": {"type": "string", "enum": ["purchase", "transfer", "payment", "withdrawal"]},
    "DeviceType": {"type": "string", "enum": ["mobile", "desktop", "ATM", "tablet"]},
    "MerchantCat": {"type": "string", "enum": ["grocery", "gas", "electronics", "travel", "clothing", "entertainment", "restaurant"]},
    "Channel": {"type": "string", "enum": ["online", "in-store", "contactless", "chip"]},
    "CardPresent": {"type": "integer", "enum": [0, 1]}
  },
  "required": ["Amount", "Age", "TxType", "DeviceType", "MerchantCat", "Channel"]
}
```

### 2. Output Schema
```json
{
  "type": "object",
  "properties": {
    "prediction": {
      "type": "integer",
      "enum": [0, 1],
      "description": "Fraud prediction (0: legitimate, 1: fraudulent)"
    },
    "probability": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Fraud probability score"
    },
    "risk_level": {
      "type": "string",
      "enum": ["low", "medium", "high"],
      "description": "Risk classification based on probability threshold"
    }
  },
  "required": ["prediction", "probability", "risk_level"]
}
```

### 3. Model Artifacts
```json
{
  "model_file": "model.pkl",
  "preprocessor": "feature_preprocessor.pkl", 
  "feature_schema": "feature_schema.json",
  "model_metadata": "model_info.json",
  "performance_report": "model_performance.html",
  "feature_importance": "feature_importance.png"
}
```

## Deployment Information

**Target Environment**: Production  
**Deployment Method**: REST API Endpoint  
**Expected Latency**: < 100ms  
**Expected Throughput**: 1000 requests/minute  
**Monitoring Requirements**: Prediction drift, performance metrics  

## Governance and Compliance

**Model Owner**: Data Science Team  
**Business Stakeholder**: Risk Management  
**Approval Status**: Pending Review  
**Compliance Requirements**: Financial Services Regulations  
**Model Validation**: Required before Production  

## Usage Instructions

1. **Input Preprocessing**: Apply feature scaling using included preprocessor
2. **Model Inference**: Use trained model for binary classification
3. **Output Interpretation**: 
   - Probability > 0.5: Flag as potential fraud
   - Probability > 0.8: High-risk fraud alert
   - Probability < 0.2: Low-risk legitimate transaction

## Model Lineage

**Training Data Source**: Domino Dataset - transformed_cc_transactions.csv  
**Feature Engineering**: exercises/c_DataEngineering/data_engineering.py  
**Training Script**: exercises/d_TrainingAndEvaluation/trainer_[model].py  
**Validation Method**: Stratified K-Fold Cross-Validation  
**Test Set Performance**: Holdout 20% validation set  

## Contact Information

**Model Developer**: Data Science Team  
**Technical Contact**: MLOps Engineering  
**Business Contact**: Risk Management Team  
**Support Email**: ml-support@company.com