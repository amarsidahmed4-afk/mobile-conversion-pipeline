# src/app.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
# ... other necessary imports (BigQuery client, pipeline utils, etc.)

app = FastAPI(title="Realtime Ecommerce Intent Engine")

# 1. Cold Boot: Load models into RAM at startup
closer_model = joblib.load("models/conversion_engine_v1.joblib")
greeter_model = joblib.load("models/greeter_engine_v1.joblib")

# 2. Define the Optuna-optimized business threshold
OPTIMAL_THRESHOLD = 0.70

class CustomerJourneyInput(BaseModel):
    # Your Pydantic schema matching the GTM payload features
    ProductRelated: int
    PageValues: float
    ExitRates: float
    # ... rest of your features

@app.post("/predict")
async def predict_intent(data: CustomerJourneyInput):
    # Convert incoming Pydantic data to DataFrame format
    input_df = preprocess_input(data) 
    
    # 3. Dynamic Routing based on behavioral depth
    if data.ProductRelated == 0:
        # Top-of-Funnel / Cold Start
        raw_probability = greeter_model.predict_proba(input_df)[0][1]
        engine_tag = "Greeter Engine"
    else:
        # Bottom-of-Funnel / Engaged User
        raw_probability = closer_model.predict_proba(input_df)[0][1]
        engine_tag = "Closer Engine"
        
    # 4. Apply the strict business threshold logic
    high_intent_flag = bool(raw_probability >= OPTIMAL_THRESHOLD)
    
    response_payload = {
        "conversion_probability": float(raw_probability),
        "high_intent_flag": high_intent_flag,
        "engine_used": engine_tag
    }
    
    # 5. Background Telemetry Log to BigQuery (Track B)
    # log_to_bigquery(row_data) 
    
    # 6. Synchronous Return to GTM (Track A)
    return response_payload