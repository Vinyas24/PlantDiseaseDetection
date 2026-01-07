import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['PYTHONHASHSEED'] = '0'

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import csv
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).parent
CORS_ORIGINS = "*"
INPUT_SIZE = 224

MODEL_REPO = "v1nyas/Plant-disease-detection-model"
MODEL_FILENAME = "plant_disease_model.keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api")

DISEASE_INFO = {}
csv_path = ROOT_DIR / "disease_info.csv"
if csv_path.exists():
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("disease_name")
            if name:
                DISEASE_INFO[name] = {
                    "description": row.get("description"),
                    "possible_steps": row.get("possible_steps"),
                    "image_url": row.get("image_url"),
                }

CLASS_NAMES = list(DISEASE_INFO.keys())
MODEL = None

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: Optional[str] = None
    possible_steps: Optional[str] = None
    image_url: Optional[str] = None

from tensorflow.keras.applications.resnet50 import preprocess_input

def load_keras_model():
    global MODEL
    try:
        logger.info("Fetching model...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        logger.info("Loading model with custom objects...")
        
        custom_objects = {
            'preprocess_input': preprocess_input
        }
        
        MODEL = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        
        gc.collect()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise e

def preprocess_image(contents: bytes) -> np.ndarray:
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32")
    arr[..., 0] -= 123.68
    arr[..., 1] -= 116.779
    arr[..., 2] -= 103.939
    
    arr = np.expand_dims(arr, axis=0)
    return arr

@api_router.get("/")
async def root():
    return {"message": "Plant Disease API is running"}

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    contents = await file.read()
    img_arr = preprocess_image(contents)

    preds = MODEL.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    predicted_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
    info = DISEASE_INFO.get(predicted_class, {})

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        description=info.get("description"),
        possible_steps=info.get("possible_steps"),
        image_url=info.get("image_url"),
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_keras_model()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
