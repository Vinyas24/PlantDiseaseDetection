import os
import gc
import logging
import uuid
import csv
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet50 import preprocess_input

class PatchedRandomFlip(tf.keras.layers.RandomFlip):
    def __init__(self, **kwargs):
        kwargs.pop('data_format', None)
        super().__init__(**kwargs)

class PatchedRandomRotation(tf.keras.layers.RandomRotation):
    def __init__(self, **kwargs):
        kwargs.pop('data_format', None)
        super().__init__(**kwargs)

class PatchedRandomZoom(tf.keras.layers.RandomZoom):
    def __init__(self, **kwargs):
        kwargs.pop('data_format', None)
        super().__init__(**kwargs)

ROOT_DIR = Path(__file__).parent
MODEL_REPO = "v1nyas/Plant-disease-detection-model"
MODEL_FILENAME = "plant_disease_model.keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISEASE_INFO = {}
CLASS_NAMES = []
MODEL = None

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: Optional[str] = None
    possible_steps: Optional[str] = None
    image_url: Optional[str] = None

def load_disease_metadata():
    global CLASS_NAMES
    csv_path = ROOT_DIR / "disease_info.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("disease_name")
                if name:
                    DISEASE_INFO[name] = row
                    CLASS_NAMES.append(name)
    logger.info(f"Loaded {len(CLASS_NAMES)} classes from CSV.")

def load_keras_model():
    global MODEL
    try:
        logger.info("Downloading model...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        custom_objects = {
            'RandomFlip': PatchedRandomFlip,
            'RandomRotation': PatchedRandomRotation,
            'RandomZoom': PatchedRandomZoom,
            'preprocess_input': preprocess_input
        }
        
        logger.info("Loading model...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            MODEL = tf.keras.models.load_model(model_path, compile=False)
        
        gc.collect()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

api_router = APIRouter(prefix="/api")

@api_router.get("/")
async def root():
    return {"status": "online", "model_loaded": MODEL is not None}

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        img_array = np.asarray(img, dtype="float32")
        img_array = np.expand_dims(img_array, axis=0)

        preds = MODEL.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        predicted_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown_{idx}"
        info = DISEASE_INFO.get(predicted_name, {})

        return PredictionResponse(
            predicted_class=predicted_name,
            confidence=confidence,
            description=info.get("description"),
            possible_steps=info.get("possible_steps"),
            image_url=info.get("image_url")
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_disease_metadata()
    load_keras_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
