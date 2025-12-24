PLANT DISEASE DETECTION USING FINE-TUNED RESNET
===============================================



This repository contains the source code, trained model, and web application for a deep learning solution designed to detect and classify various plant leaf diseases. The core model uses a **Fine-Tuned ResNet Architecture** for powerful image feature extraction.

***

FEATURES
--------

* **Model Architecture:** Utilizes a **Fine-Tuned ResNet** (e.g., ResNet50) for robust classification.
* **Deployment:** Lightweight, high-performance web service built with **FastAPI** and **Uvicorn**.
* **Disease Identification:** Capable of classifying **N** distinct disease categories and healthy leaves.
* **Database Integration:** Loads disease information, descriptions, and possible steps from a `disease_info.csv` file.

***

TECHNOLOGY STACK
----------------

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model** | **TensorFlow / Keras** | Core ML framework for model loading and prediction. |
| **Backend API** | **FastAPI** | Python framework for the RESTful prediction endpoint. |
| **Server** | **Uvicorn** | ASGI server for high-speed API execution. |
| **Dependencies** | **`dotenv`, `Pydantic`** | Environment management and data validation. |
| **Image Handling** | **PIL (Pillow), NumPy** | Preprocessing leaf images for model input. |

***

INSTALLATION AND SETUP
----------------------

### 1. Clone the Repository

```bash
git clone [https://github.com/Vinyas24/PlantDiseaseDetection.git](https://github.com/Vinyas24/PlantDiseaseDetection.git)
cd PlantDiseaseDetection
```

### 2. Backend Environment Setup

Navigate to the `backend` directory and install dependencies.

```bash
cd backend

# Create and activate a Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install all required Python packages (requires a requirements.txt file)
pip install -r requirements.txt
```

### 3. Model File

Ensure your trained Keras model (`model.h5`) and the supplementary data file (`disease_info.csv`) are located in the **`backend/`** directory.

> **Note:** This repository uses **Git LFS** to handle the large `model.h5` file. Please ensure Git LFS is installed (`sudo apt install git-lfs` and `git lfs install`).

***

RUNNING THE APPLICATION
-----------------------

### 1. Start the Backend API

Run the Uvicorn server from the **`backend`** directory (with your `venv` active):

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://0.0.0.0:8000`.

### 2. Run the Frontend (Client)

Navigate to the `frontend` directory and start your client application (assuming a standard Node.js setup):

```bash
cd ../frontend
npm install  # or yarn install
npm start    # or yarn start
```

The frontend (typically at `http://localhost:3000`) will connect to the backend to get disease predictions.

***

API ENDPOINT
------------

The primary endpoint for classification is:

### POST /api/predict

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`file`** | `UploadFile` | The leaf image file to be classified. |

**Success Response Example:**

```json
{
  "predicted_class": "Tomato_Bacterial_Spot",
  "confidence": 0.985,
  "description": "Caused by the bacterium Xanthomonas...",
  "possible_steps": "Apply copper fungicides...",
  "image_url": "[http://example.com/tomato_spot.jpg](http://example.com/tomato_spot.jpg)"
}
```

***

MODEL DETAILS
-------------

The model load logic is configured to handle potential security warnings related to custom Python code (like `Lambda` layers) often present in Keras H5 files:

```python
# Fix applied in server.py to bypass security errors:
MODEL = load_model(str(model_path), compile=False, safe_mode=False)
```

***

CONTRIBUTING
------------

We welcome contributions, bug reports, and suggestions! Please feel free to open an issue or submit a pull request.

***
