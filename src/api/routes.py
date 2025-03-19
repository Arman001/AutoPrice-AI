# src/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services import predict_price

router = APIRouter(prefix="/api", tags=["Prediction"])

# Define request model
class CarFeatures(BaseModel):
    year: int
    mileage: float
    brand: str
    fuel_type: str
    transmission: str
    engine: float  # Added engine size
    ext_col: str   # Added exterior color
    int_col: str   # Added interior color
    accident: int  # Added accident count

@router.post("/predict")
def predict(features: CarFeatures):
    try:
        prediction = predict_price(features.dict())

        # Convert NumPy float to Python float
        prediction = round(float(prediction),2)

        # print("✅ Prediction:", prediction)
        return {"predicted_price": prediction}
    except Exception as e:
        print("❌ Prediction Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
