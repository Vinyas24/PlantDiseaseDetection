import os
import gc
import logging
import csv
from pathlib import Path
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

tf.keras.utils.get_custom_objects().update({
    'RandomFlip': PatchedRandomFlip,
    'RandomRotation': PatchedRandomRotation,
    'RandomZoom': PatchedRandomZoom,
    'preprocess_input': preprocess_input
})

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

def load_keras_model():
    global MODEL
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        MODEL = tf.keras.models.load_model(model_path, compile=False)
        gc.collect()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise e

api_router = APIRouter(prefix="/api")

@api_router.get("/health")
async def health():
    return {"status": "ok", "model_ready": MODEL is not None}

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model loading...")
    
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)

    preds = MODEL.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    
    name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown_{idx}"
    info = DISEASE_INFO.get(name, {})

    return PredictionResponse(
        predicted_class=name,
        confidence=float(preds[idx]),
        description=info.get("description"),
        possible_steps=info.get("possible_steps"),
        image_url=info.get("image_url")
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_disease_metadata()
    load_keras_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
