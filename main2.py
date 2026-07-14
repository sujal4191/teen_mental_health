import pickle
from fastapi import FastAPI,HTTPException
import pandas as pd
from pydantic import BaseModel

model=pickle.load(open("mental_health_model.pkl", "rb"))
encoders=pickle.load(open("encoders.pkl", "rb"))
app=FastAPI(
    title="Teen Mental Health Prediction API",
    description="Predicts whether a teenager is likely to be depressed based on lifestyle and social media usage.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Welcome to the Teen Mental Health Prediction API!",
        "status": "API is running successfully."
    }

@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model": "Random Forest Classifier",
        "prediction": "Depression Detection"
    }

class Teendata(BaseModel):
    age: int
    gender: str
    daily_social_media_hours: float
    platform_usage: str
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: str
    stress_level: float
    anxiety_level: float
    addiction_level: float

@app.post("/predict")
def predict(data: Teendata):

    try:

        input_data = {
            "age": data.age,
            "gender": data.gender,
            "daily_social_media_hours": data.daily_social_media_hours,
            "platform_usage": data.platform_usage,
            "sleep_hours": data.sleep_hours,
            "screen_time_before_sleep": data.screen_time_before_sleep,
            "academic_performance": data.academic_performance,
            "physical_activity": data.physical_activity,
            "social_interaction_level": data.social_interaction_level,
            "stress_level": data.stress_level,
            "anxiety_level": data.anxiety_level,
            "addiction_level": data.addiction_level
        }

        df = pd.DataFrame([input_data])

        # Normalize input
        df["gender"] = df["gender"].str.strip().str.lower()
        df["platform_usage"] = df["platform_usage"].str.strip().str.capitalize()
        df["social_interaction_level"] = df["social_interaction_level"].str.strip().str.lower()

        # Encode categorical columns
        for column in ["gender", "platform_usage", "social_interaction_level"]:
            df[column] = encoders[column].transform(df[column])

        # Prediction
        y_pred = model.predict(df)[0]

        probability = model.predict_proba(df)[0]
        confidence= round(max(probability) * 100, 2)

        if y_pred == 1:
            status = "Depression Detected"

            if confidence >= 90:
                risk = "High"
            else:
                risk = "Medium"

            recommendation = (
                "Consult a mental health professional. Improve sleep, "
                "reduce excessive social media usage, exercise regularly, "
                "and seek emotional support."
    )
        else:
            status = "No Depression Detected"
            risk = "Low"
            recommendation = (
                "maintain a healthy lifestyle,balance sreen time, engage in physical activities, and seek support if needed."
            )
        return {
            "prediction": "depression" if y_pred == 1 else "No depression",
            "status": status,
            "risk_level": risk,
            "confidence": f"{confidence}%",
            "recommendation": recommendation
        }    

    

    

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )