import os
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

def load_keras_model():
    global MODEL
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    MODEL = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Model loaded successfully")

def preprocess_image(contents: bytes) -> np.ndarray:
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype="float32")
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status = StatusCheck(**input.model_dump())
    STATUS_STORE.append(status.model_dump())
    return status

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    return STATUS_STORE

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    contents = await file.read()
    img_arr = preprocess_image(contents)

    preds = MODEL.predict(img_arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    predicted_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
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
