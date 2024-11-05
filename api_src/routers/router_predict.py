from fastapi import APIRouter
from api_src.logger.logger import get_logger
from api_src.config.settings import get_settings
from api_src.schemas.schema_predict import PredictAllRequest , PredictAttackRequest , PredictAttackCatRequest
from fastapi import UploadFile, File
from fastapi import APIRouter, HTTPException
import pandas as pd
from io import BytesIO
from api_src.services.service_predict import ServicePredict
from src.models.UNSW_NB15_models.CatModel import CatModel
import numpy as np 
model = CatModel()
service_instance = ServicePredict(model = model )


logger = get_logger(__file__)
settings = get_settings()
router = APIRouter()



@router.post(path="/predict-all")
async def predict_all(predict_all_request : PredictAllRequest): 
    pass 


@router.post(path="/predict-attack-cat" ) 
async def predict_attack_cat(predict_attack_cat_request : PredictAttackCatRequest): 
    pass 


@router.post("/predict-attack")
async def predict_attack(file: UploadFile = File(...)):
    content = await file.read()
    parquet_buffer = BytesIO(content)
    
    try:
        features = pd.read_parquet(parquet_buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read Parquet file.")

    try:
        preds = await service_instance.apredict_detection(features)
        
        # Ensure preds is in JSON-compatible format
        if isinstance(preds, pd.DataFrame):
            preds = preds.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
        elif isinstance(preds, np.ndarray):
            preds = preds.tolist()  # Convert numpy array to list
        elif isinstance(preds, dict):
            preds = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in preds.items()}

        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed.")
