from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional

# **NEW**: Import metrics from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- MOCK FUNCTIONS (for standalone testing if utils are not available) ---
# Replace these with your actual imports if you have them in a different structure
def load_object(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(error)
        print(f"Error: {error} in {error_detail}")
# --- END MOCK FUNCTIONS ---

# Pydantic models (no changes needed here)
class URLFeatures(BaseModel):
    having_IP_Address: int; URL_Length: int; Shortining_Service: int; having_At_Symbol: int; double_slash_redirecting: int; Prefix_Suffix: int; having_Sub_Domain: int; SSLfinal_State: int; Domain_registeration_length: int; Favicon: int; port: int; HTTPS_token: int; Request_URL: int; URL_of_Anchor: int; Links_in_tags: int; SFH: int; Submitting_to_email: int; Abnormal_URL: int; Redirect: int; on_mouseover: int; RightClick: int; popUpWidnow: int; Iframe: int; age_of_domain: int; DNSRecord: int; web_traffic: int; Page_Rank: int; Google_Index: int; Links_pointing_to_page: int; Statistical_report: int

class PredictionResponse(BaseModel):
    prediction: int; confidence: Optional[float] = None; risk_level: str; message: str

class NetworkModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor; self.model = model
    def predict(self, x):
        x_transform = self.preprocessor.transform(x); return self.model.predict(x_transform)
    def predict_proba(self, x):
        if hasattr(self.model, 'predict_proba'):
            x_transform = self.preprocessor.transform(x); return self.model.predict_proba(x_transform)
        return None

# Initialize FastAPI app
app = FastAPI(title="URL Safety Checker API", version="1.2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])

templates = Jinja2Templates(directory="templates")
os.makedirs("prediction_output", exist_ok=True)
app.mount("/prediction_output", StaticFiles(directory="prediction_output"), name="prediction_output")

network_model = None

@app.on_event("startup")
def load_model():
    global network_model
    try:
        preprocessor = load_object("final_model/preprocessing.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def get_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy" if network_model else "degraded", "model_loaded": network_model is not None}

@app.post("/predict_url", response_model=PredictionResponse)
async def predict_url_safety(features: URLFeatures):
    if not network_model: raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        df = pd.DataFrame([features.model_dump()]); prediction = int(network_model.predict(df)[0])
        confidence = 0.5; probabilities = network_model.predict_proba(df)
        if probabilities is not None: confidence = float(probabilities[0][prediction])
        if prediction == 1: risk_level, message = "HIGH", "This URL shows strong characteristics of a phishing/malicious website."
        else: risk_level, message = "LOW", "This URL appears to be legitimate and safe to visit."
        return PredictionResponse(prediction=prediction, confidence=confidence, risk_level=risk_level, message=message)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict")
async def predict_csv_and_show_table(request: Request, file: UploadFile = File(...)):
    """
    Accepts a CSV file. If it contains 'predicted_column' (as ground truth),
    it evaluates the model's performance. Otherwise, it just returns predictions.
    """
    if not network_model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        df = pd.read_csv(file.file)
        evaluation_results = None
        
        # --- NEW: EVALUATION MODE LOGIC ---
        if 'predicted_column' in df.columns:
            # 1. Separate features from the ground truth
            y_true = df['predicted_column'].astype(int)
            # Ensure the ground truth column is not sent to the model for prediction
            features_df = df.drop(columns=['predicted_column'])
            
            # Make predictions on the features
            y_pred = network_model.predict(features_df).astype(int)
            
            # 2. Calculate performance metrics
            # Use labels=[0, 1] to handle cases where a batch might only contain one class
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            evaluation_results = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
            }
            
            # 3. Prepare DataFrame for display
            # Rename the original column for clarity and add the model's prediction
            df.rename(columns={'predicted_column': 'Actual_Result'}, inplace=True)
            df['Model_Prediction'] = y_pred
            
        else: # If no ground truth column, perform prediction only
            y_pred = network_model.predict(df).astype(int)
            df['Model_Prediction'] = y_pred

        # Save results to a CSV file
        output_path = os.path.join('prediction_output', 'output.csv')
        df.to_csv(output_path, index=False)

        # Generate HTML table for display
        table_html = df.to_html(classes='table table-striped table-hover', index=False)
        
        # Render the results, including evaluation metrics if they exist
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html, "evaluation": evaluation_results}
        )

    except Exception as e:
        # Catch potential errors like missing feature columns in the uploaded CSV
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enhanced_api:app", host="0.0.0.0", port=8000, reload=True)