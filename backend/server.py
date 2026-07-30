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

# --- MEMORY OPTIMIZATIONS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Sometimes saves RAM by avoiding specialized kernels
try:
    import keras
    keras.config.set_floatx('float16') # Significantly reduces RAM usage for inference
    logging.info("Keras floatx set to float16 for memory efficiency.")
except:
    pass

# --- THE NUCLEAR PATCH ---
# We intercept Keras deserialization at its core to strip out any arguments
# that cause version compatibility failures (Keras 2 -> 3).
try:
    import keras
    from keras.src.saving import serialization_lib
    from keras.src.legacy.saving import serialization as legacy_serialization

    def make_patched_deserialize(orig_deserialize):
        def patched_deserialize(config, custom_objects=None, **kwargs):
            def sanitize_config(obj):
                if isinstance(obj, dict):
                    # Map Keras 2 input shape to Keras 3 brand
                    if 'batch_input_shape' in obj:
                        obj['batch_shape'] = obj.pop('batch_input_shape')

                    # Remove problematic Keras 2/3 mismatch keys
                    problematic = [
                        'quantization_config', 'data_format', 'mode', 'factor', 
                        'height_factor', 'width_factor', 'fill_mode', 
                        'interpolation', 'seed', 'fill_value'
                    ]
                    for key in list(obj.keys()):
                        if key in problematic:
                            obj.pop(key, None)
                    for val in obj.values():
                        sanitize_config(val)
                elif isinstance(obj, list):
                    for item in obj:
                        sanitize_config(item)

            if isinstance(config, dict):
                sanitize_config(config)
                # Ensure the class_name mapping is respected
                if custom_objects:
                    cls_name = config.get('class_name')
                    if cls_name in custom_objects:
                        config['module'] = '__main__'

            return orig_deserialize(config, custom_objects=custom_objects, **kwargs)

        return patched_deserialize

    serialization_lib.deserialize_keras_object = make_patched_deserialize(
        serialization_lib.deserialize_keras_object
    )
    legacy_serialization.deserialize_keras_object = make_patched_deserialize(
        legacy_serialization.deserialize_keras_object
    )
    logging.info("Keras deserialization nuclear patch armed.")
except Exception as e:
    logging.warning(f"Nuclear patch failed to initialize: {e}")

# Preprocessing layers are identity during inference
class BypassLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__()
    def call(self, inputs): return inputs
    @classmethod
    def from_config(cls, config): return cls()

CUSTOM_OBJECTS = {
    'RandomFlip': BypassLayer,
    'RandomRotation': BypassLayer,
    'RandomZoom': BypassLayer,
    'Sequential': tf.keras.Sequential,
}

ROOT_DIR = Path(__file__).parent
MODEL_REPO = "v1nyas/plant-disease-detection"
MODEL_FILENAME = "plant_disease_model.h5"

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
        # Clear any existing state
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Limit TF thread usage to reduce memory overhead
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        logger.info("Downloading model from HF...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        
        logger.info(f"Loading model (float16) from {model_path}...")
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
    img_array = np.array(img).astype('float16') 
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = MODEL.predict(img_array, verbose=0)[0]
    
    # Cleanup after prediction to prevent RAM bloat
    del img_array
    gc.collect()

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
