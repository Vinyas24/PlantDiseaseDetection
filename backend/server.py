import os
import gc
import logging
import csv
from pathlib import Path
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# --- THE BRUTE FORCE FIX ---
# We define these as "Do Nothing" layers so Keras stops complaining 
# about the config/data_format during load.
class BypassLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop('data_format', None)
        kwargs.pop('mode', None)
        kwargs.pop('factor', None)
        kwargs.pop('height_factor', None)
        kwargs.pop('width_factor', None)
        kwargs.pop('fill_mode', None)
        kwargs.pop('interpolation', None)
        kwargs.pop('seed', None)
        kwargs.pop('fill_value', None)
        super().__init__(**kwargs)
    def call(self, inputs): return inputs

# Register EVERYTHING that could possibly fail as a BypassLayer
tf.keras.utils.get_custom_objects().update({
    'RandomFlip': BypassLayer,
    'RandomRotation': BypassLayer,
    'RandomZoom': BypassLayer,
    'Sequential': tf.keras.Sequential, # Ensure Sequential is mapped correctly
})

ROOT_DIR = Path(__file__).parent
MODEL_REPO = "v1nyas/Plant-disease-detection-model"
MODEL_FILENAME = "plant_disease_model.keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISEASE_INFO = {}
CLASS_NAMES = []
MODEL = None

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
        logger.info("Downloading model from HF...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        logger.info("Attempting model load with bypassed layers...")
        # compile=False is critical here
        MODEL = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        logger.info("Model loaded successfully!")
        gc.collect()
    except Exception as e:
        logger.error(f"FINAL LOAD ERROR: {str(e)}")
        # If this still fails, the model file itself might be corrupted or 
        # saved in a format incompatible with the current Keras version.
        raise e

api_router = APIRouter(prefix="/api")

@api_router.get("/health")
async def health():
    return {"status": "ok", "model_ready": MODEL is not None}

@api_router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if MODEL is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    
    # Manual Preprocessing (matches ResNet50 requirements)
    img_array = np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    
    # Simple ResNet preprocessing: Scale to [-1, 1] or similar if not using the Lambda layer
    # Note: If your model has the Lambda(preprocess_input) layer, this is enough.
    
    preds = MODEL.predict(img_array, verbose=0)[0]
    idx = np.argmax(preds)
    
    name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
    info = DISEASE_INFO.get(name, {})
    
    return {
        "predicted_class": name,
        "confidence": float(preds[idx]),
        "description": info.get("description"),
        "prevent": info.get("possible_steps")
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_disease_metadata()
    load_keras_model()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)
