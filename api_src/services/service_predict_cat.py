from src.models.UNSW_NB15_models.ClassificationModel import ClassificationModel
import pandas as pd 
from api_src.logger.logger import get_logger

logger = get_logger(__file__)

class ServicePredictCat() : 
    def __init__(self, model : ClassificationModel) :
        self.model = model 

    async def apredict_classification(self , features : pd.DataFrame ) : 
        try : 
            preds = self.model.predict(features)
            return preds
        except Exception as e : 
            logger.error(f"Error occured in service_predict.apredict_attack : {e}")