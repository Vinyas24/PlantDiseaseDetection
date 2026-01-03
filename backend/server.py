import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
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

try:
    import tensorflow as tf
    from tensorflow.keras.applications import imagenet_utils
    from tensorflow.keras.models import load_model
    import keras

    try:
        keras.config.enable_unsafe_deserialization()
    except AttributeError:
        pass
    
except ImportError:
    tf = None
    load_model = None
    imagenet_utils = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
MODEL_FILE = os.environ.get("MODEL_FILE", str(ROOT_DIR / "plant_disease_model_augmented.h5"))
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 224))
CLASS_NAMES_FILE = os.environ.get("CLASS_NAMES_FILE", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api")

DISEASE_INFO = {}
try:
    csv_path = ROOT_DIR / "disease_info.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                disease_name = row.get("disease_name")
                if not disease_name:
                    continue
                DISEASE_INFO[disease_name] = {
                    "description": row.get("description"),
                    "possible_steps": row.get("Possible Steps") or row.get("possible_steps"),
                    "image_url": row.get("image_url"),
                }
except Exception as e:
    logger.error(f"Error loading disease_info.csv: {str(e)}")
    DISEASE_INFO = {}

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: Optional[str] = None
    possible_steps: Optional[str] = None
    image_url: Optional[str] = None

STATUS_STORE: List[dict] = []
PREDICTIONS_STORE: List[dict] = []
MODEL = None
MODEL_LOAD_ERROR = None 
CLASS_NAMES: List[str] = []

def load_class_names():
    global CLASS_NAMES
    if CLASS_NAMES_FILE:
        p = Path(CLASS_NAMES_FILE)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
                CLASS_NAMES = class_names
            return

    if DISEASE_INFO:
        CLASS_NAMES = list(DISEASE_INFO.keys())
        return

def load_keras_model():
    global MODEL, MODEL_LOAD_ERROR
    if load_model is None:
        MODEL_LOAD_ERROR = "TensorFlow/Keras not installed."
        logger.error(MODEL_LOAD_ERROR)
        return

    model_path = Path(MODEL_FILE)
    logger.info(f"Looking for model at: {model_path.absolute()}")
    
    if not model_path.exists():
        MODEL_LOAD_ERROR = f"File not found at {model_path.absolute()}"
        logger.error(MODEL_LOAD_ERROR)
        return

    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        MODEL = load_model(
            str(model_path),
            compile=False,
            custom_objects={"<lambda>": preprocess_input, "imagenet_utils": imagenet_utils}
        )
        logger.info("Model loaded successfully")
        MODEL_LOAD_ERROR = None
    except Exception as e:
        MODEL_LOAD_ERROR = f"Keras Load Error: {str(e)}"
        logger.error(f"Failed to load model: {e}", exc_info=True)
        MODEL = None

def preprocess_image(contents: bytes, input_size: int = INPUT_SIZE) -> np.ndarray:
    img = Image.open(BytesIO(contents))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((input_size, input_size), Image.LANCZOS) 
    arr = np.asarray(img).astype("float32") 
    arr = np.expand_dims(arr, axis=0)
    return arr

@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(**input.model_dump())
    doc = status_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    STATUS_STORE.append(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    out = []
    for s in STATUS_STORE:
        s_copy = s.copy()
        if isinstance(s_copy.get("timestamp"), str):
            s_copy["timestamp"] = datetime.fromisoformat(s_copy["timestamp"])
        out.append(s_copy)
    return out

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if MODEL is None:
            detail_msg = f"Model not loaded. Reason: {MODEL_LOAD_ERROR}"
            logger.error(detail_msg)
            raise HTTPException(status_code=500, detail=detail_msg)

        try:
            input_arr = preprocess_image(contents, INPUT_SIZE)
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise HTTPException(status_code=400, detail="Error preprocessing image")

        try:
            preds = MODEL.predict(input_arr)
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            raise HTTPException(status_code=500, detail="Error during model prediction")

        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
        
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        if CLASS_NAMES and class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[class_idx]
        else:
            predicted_class = str(class_idx)

        disease_info = DISEASE_INFO.get(predicted_class, {})

        prediction_record = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": disease_info.get("description"),
            "possible_steps": disease_info.get("possible_steps"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        PREDICTIONS_STORE.append(prediction_record)

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            description=disease_info.get("description"),
            possible_steps=disease_info.get("possible_steps"),
            image_url=disease_info.get("image_url"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error in /predict: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    load_class_names()
    load_keras_model()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ORIGINS.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
