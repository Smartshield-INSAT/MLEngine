from pydantic import BaseModel
from api_src.logger.logger import get_logger
from fastapi import UploadFile


logger = get_logger(__file__)

class PredictAllRequest(BaseModel) : 
    pass

class PredictAttackCatRequest(BaseModel) : 
    file : UploadFile 


class PredictAttackRequest(BaseModel):
    file : UploadFile  # Add more fields if needed
