from fastapi import APIRouter
from api_src.logger.logger import get_logger
from api_src.config.settings import get_settings
from api_src.schemas.schema_predict import PredictAllRequest , PredictAttackRequest , PredictAttackCatRequest
logger = get_logger(__file__)
settings = get_settings()
router = APIRouter()



@router.post(path="/predict-all")
async def predict_all(predict_all_request : PredictAllRequest): 
    pass 


@router.post(path="/predict-attack-cat" ) 
async def predict_attack_cat(predict_attack_cat_request : PredictAttackCatRequest): 
    pass 

@router.post(path="/predict-attack")
async def predict_attack(predict_attack : PredictAttackRequest): 
    pass 