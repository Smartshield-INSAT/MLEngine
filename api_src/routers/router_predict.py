from fastapi import APIRouter
from api_src.logger.logger import get_logger
from api_src.config.settings import get_settings
from api_src.schemas.schema_predict import PredictAllRequest , PredictAttackRequest , PredictAttackCatRequest
from fastapi import UploadFile, File
from fastapi import APIRouter, HTTPException
import pandas as pd
from io import BytesIO
from api_src.services.service_predict import ServicePredict
from src.models.UNSW_NB15_models.DetectionModel import DetectionModel
import numpy as np 
from api_src.services.service_predict_cat import ServicePredictCat
from src.models.UNSW_NB15_models.ClassificationModel import ClassificationModel
detection_model = DetectionModel()
detection_service_instance = ServicePredict(model = detection_model)
classification_model = ClassificationModel()
classification_service_instance = ServicePredictCat(model = classification_model)


logger = get_logger(__file__)
settings = get_settings()
router = APIRouter()



@router.post(path="/predict-all")
async def predict_all(file: UploadFile = File(...)): 
    content = await file.read()
    parquet_buffer = BytesIO(content)

    try :
        features = pd.read_parquet(parquet_buffer)
    except Exception as e :
        logger.error(f"Failed to read Parquet file.{e}")
        raise HTTPException(status_code=400 , detail="Failed to read Parquet file.")

    try :
        preds = await detection_service_instance.apredict_detection(features)
        
        # Ensure preds is in JSON-compatible format
        if isinstance(preds, pd.DataFrame):
            preds = preds.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
        elif isinstance(preds, np.ndarray):
            preds = preds.tolist()  # Convert numpy array to list
        elif isinstance(preds, dict):
            preds = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in preds.items()}

        logger.debug(f"Predictions : {preds}")
        preds_cat = await classification_service_instance.apredict_classification(features)
        
        # Ensure preds is in JSON-compatible format
        if isinstance(preds_cat, pd.DataFrame):
            preds_cat = preds_cat.to_dict(orient="records")

        elif isinstance(preds_cat, np.ndarray):
            preds_cat = preds_cat.tolist()

        elif isinstance(preds_cat, dict):
            preds_cat = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in preds_cat.items()}

        logger.debug(f"Predictions : {preds_cat}")

        return [pred_cat * pred for pred , pred_cat in zip(preds , preds_cat)]

    except Exception as e :
        logger.error(f"Prediction failed.{e}")

@router.post(path="/predict-attack-cat" ) 
async def predict_attack_cat(file : UploadFile = File(...)) : 
    content = await file.read()
    parquet_buffer = BytesIO(content)

    try :
        features = pd.read_parquet(parquet_buffer)
    except Exception as e :
        raise HTTPException(status_code=400 , detail="Failed to read Parquet file.")

    try :
        preds = await classification_service_instance.apredict_classification(features)
        
        # Ensure preds is in JSON-compatible format
        if isinstance(preds, pd.DataFrame):
            preds = preds.to_dict(orient="records")  # Convert DataFrame to a list of dictionaries
        elif isinstance(preds, np.ndarray):
            preds = preds.tolist()  # Convert numpy array to list
        elif isinstance(preds, dict):
            preds = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in preds.items()}

        return {"predictions": preds}
    except Exception as e :
        raise HTTPException(status_code=500 , detail="Prediction failed.")
    


@router.post(path="/predict-attack")
async def predict_attack(file: UploadFile = File(...)):
    content = await file.read()
    parquet_buffer = BytesIO(content)
    
    try:
        features = pd.read_parquet(parquet_buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read Parquet file.")

    try:
        preds = await detection_service_instance.apredict_detection(features)
        
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
