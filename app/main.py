from fastapi import FastAPI
from pydantic import BaseModel
from app.ai import call_model
from app.datacreate import to_numpy
from app.weather import fetch_weather
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = await call_model(request.message)
    return ChatResponse(reply=reply)

@app.get("/predict")
async def predict(lat: float, lon: float):
    data  = await fetch_weather(lat, lon)
    dates, values = to_numpy(data)

    logger.info(dates)
    logger.info(values)


    # return data
