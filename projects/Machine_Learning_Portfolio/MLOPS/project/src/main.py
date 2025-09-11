import logging
import os

from typing import List
from redis import asyncio
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis


model_path = "./distilbert-base-uncased-finetuned-sst2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    return_all_scores=True,    
)

logger = logging.getLogger(__name__)
LOCAL_REDIS_URL = "redis://redis:6379"
app = FastAPI()


@app.on_event("startup")
def startup():    
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    
    # redis = aioredis.from_url(LOCAL_REDIS_URL, encoding="utf8", decode_responses=True)
    # FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

class SentimentRequest(BaseModel):
    text: list[str]


class Sentiment(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    predictions: list[list[Sentiment]]


@app.post("/predict", response_model=SentimentResponse)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):  # Update to async function
    return {"predictions": classifier(sentiments.text)}


@app.get("/health")
async def health():
    return {"status": "healthy"}
