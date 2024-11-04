import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from api_src.config.settings import get_settings
from api_src.logger.logger import get_logger
settings = get_settings()
logger = get_logger(__file__)


ascii_art ="""

░░      ░░░  ░░░░  ░░░      ░░░       ░░░        ░░░      ░░░  ░░░░  ░░        ░░        ░░  ░░░░░░░░       ░░
▒  ▒▒▒▒▒▒▒▒   ▒▒   ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒
▓▓      ▓▓▓        ▓▓  ▓▓▓▓  ▓▓       ▓▓▓▓▓▓  ▓▓▓▓▓▓      ▓▓▓        ▓▓▓▓▓  ▓▓▓▓▓      ▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓  ▓
███████  ██  █  █  ██        ██  ███  ██████  ███████████  ██  ████  █████  █████  ████████  ████████  ████  █
██      ███  ████  ██  ████  ██  ████  █████  ██████      ███  ████  ██        ██        ██        ██       ██
                                                                                                              

"""
app = FastAPI(
    title="AI API App SMARTSHIELD",
)

logger.info(f"Starting App : \n {ascii_art}")

logger.info("App Ready")

@app.get("/", response_class=PlainTextResponse)
async def root():
    return ascii_art



if __name__ == "__main__":
    try : 
        uvicorn.run(
            app,
            port=8002,
            host="0.0.0.0",

        )
    except KeyboardInterrupt as ki : 
        logger.info("Turning Server Off ...")
        logger.info("server Off")
    except Exception as e : 
        logger.error(f"Error occured in app : {e}")
        
    
