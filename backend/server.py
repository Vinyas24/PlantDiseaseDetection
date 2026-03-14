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
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# --- THE BRUTE FORCE FIX (v2) ---
# We use a decorator to ensure Keras 3's serialization system finds our compatible versions.
def make_compatible(cls, name):
    if cls is None: return BypassLayer
    
    @tf.keras.utils.register_keras_serializable(package="Compatibility", name=name)
    class CompatibleLayer(cls):
        def __init__(self, **kwargs):
            # Strip offending Keras 2/3 mismatch arguments
            offenders = [
                'data_format', 'mode', 'factor', 'height_factor', 'width_factor', 
                'fill_mode', 'interpolation', 'seed', 'fill_value', 'batch_shape',
                'sparse', 'ragged', 'quantization_config', 'batch_input_shape'
            ]
            for arg in offenders: kwargs.pop(arg, None)
            super().__init__(**kwargs)
        
        @classmethod
        def from_config(cls, config):
            # Deep-clean the config before init
            for k in ['module', 'class_name', 'registered_name']: config.pop(k, None)
            return cls(**config)
            
    # Also patch the __name__ to match original for some legacy loaders
    CompatibleLayer.__name__ = cls.__name__
    return CompatibleLayer

@tf.keras.utils.register_keras_serializable(package="Compatibility", name="BypassLayer")
class BypassLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__()
    def call(self, inputs): return inputs
    @classmethod
    def from_config(cls, config): return cls()

# Map EVERYTHING to catch all possible deserialization paths
CUSTOM_OBJECTS = {
    'RandomFlip': BypassLayer,
    'RandomRotation': BypassLayer,
    'RandomZoom': BypassLayer,
    'InputLayer': BypassLayer,
    'Dense': make_compatible(tf.keras.layers.Dense, 'Dense'),
    'Conv2D': make_compatible(tf.keras.layers.Conv2D, 'Conv2D'),
    'BatchNormalization': make_compatible(tf.keras.layers.BatchNormalization, 'BatchNormalization'),
    'Activation': make_compatible(tf.keras.layers.Activation, 'Activation'),
    'MaxPooling2D': make_compatible(tf.keras.layers.MaxPooling2D, 'MaxPooling2D'),
    'ZeroPadding2D': make_compatible(tf.keras.layers.ZeroPadding2D, 'ZeroPadding2D'),
    'GlobalAveragePooling2D': make_compatible(tf.keras.layers.GlobalAveragePooling2D, 'GlobalAveragePooling2D'),
    'Flatten': make_compatible(tf.keras.layers.Flatten, 'Flatten'),
    'Add': make_compatible(tf.keras.layers.Add, 'Add'),
    'Sequential': tf.keras.Sequential,
}

# Apply globally as well
tf.keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

ROOT_DIR = Path(__file__).parent
MODEL_REPO = "v1nyas/plant-disease-detection"
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
        # compile=False is critical here. 
        # We use custom_object_scope to ensure even nested layers are intercepted.
        with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            MODEL = tf.keras.models.load_model(
                model_path, 
                compile=False, 
                safe_mode=False
            )
        
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
    img_array = preprocess_input(img_array)
    
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
