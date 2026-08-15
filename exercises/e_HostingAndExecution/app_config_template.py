# Fraud Detection App Configuration Template
# Copy this file to app_config.py and replace the placeholder values with your actual endpoints

# Feature Scaling Endpoint Configuration
FEATURE_SCALING_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-feature-scaling-model-id}/latest/model"
FEATURE_SCALING_AUTH = "your-auth-token-here"

# Model Endpoints Configuration
XGBOOST_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-xgboost-model-id}/latest/model"
XGBOOST_AUTH = "your-auth-token-here"

ADABOOST_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-adaboost-model-id}/latest/model"
ADABOOST_AUTH = "your-auth-token-here"

GAUSSIANNB_ENDPOINT = "https://se-demo.domino.tech:443/models/{your-gaussiannb-model-id}/latest/model"
GAUSSIANNB_AUTH = "your-auth-token-here"

# Model Configuration Dictionary
MODEL_CONFIG = {
    'XG Boost': {
        'endpoint': XGBOOST_ENDPOINT,
        'auth': XGBOOST_AUTH,
    },
    'ADA Boost': {
        'endpoint': ADABOOST_ENDPOINT,
        'auth': ADABOOST_AUTH,
    },
    'GaussianNB': {
        'endpoint': GAUSSIANNB_ENDPOINT,
        'auth': GAUSSIANNB_AUTH,
    }
}