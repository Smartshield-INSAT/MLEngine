from pydantic import BaseModel, Field
from typing import List , Optional 
from abc import ABC
from api_src.logger.logger import get_logger


logger = get_logger(__file__)

class PredictAllRequest(BaseModel) : 
    pass

class PredictAttackCatRequest(BaseModel) : 
    pass 

class PredictAttackRequest(BaseModel) : 
    pass 

