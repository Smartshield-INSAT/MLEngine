from src.models.UNSW_NB15_models.DetectionModel import DetectionModel
import pandas as pd 
from api_src.logger.logger import get_logger

logger = get_logger(__file__)


class ServicePredict() : 
    def __init__(self, model : DetectionModel) :
        self.model = model 

    async def apredict_detection(self , features : pd.DataFrame ) : 
        try : 
            preds = self.model.predict(features)
            return preds
        except Exception as e : 
            logger.error(f"Error occured in service_predict.apredict_attack : {e}")

        
        