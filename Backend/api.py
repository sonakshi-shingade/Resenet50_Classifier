from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# 🟢 Map of model name → file path
MODEL_PATHS = {
    "bike_car_model": "resnet50_car_bike.h5",
    # "truck_car_model": "truck_car_model.h5"
}

# 🟢 Map of model name → loaded model (load once at startup)
MODELS = {}
for model_name, file_path in MODEL_PATHS.items():
    try:
        MODELS[model_name] = load_model(file_path)
        logger.info(f"✅ Loaded model: {model_name} from {file_path}")
    except Exception as e:
        logger.error(
            f"⚠️ Could not load model {model_name} from {file_path}: {e}")

# 🟢 Define class names for each model
CLASS_NAMES = {
    "bike_car_model": ["bike", "car"],
    "truck_car_model": ["truck", "car"]
}


class PredictRequest(BaseModel):
    model: str
    encoded_image: str


def preprocess_image(image_bytes):
    logger.info("Preprocessing image...")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Adjust if needed
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    logger.info("Image preprocessing completed.")
    return img_array


@app.post("/predict/")
async def predict(request: PredictRequest):
    model_key = request.model
    logger.info(f"Received prediction request for model: {model_key}")

    if model_key not in MODELS:
        logger.warning(f"Requested model not found: {model_key}")
        raise HTTPException(
            status_code=400, detail=f"Model '{model_key}' not found.")

    try:
        image_bytes = base64.b64decode(request.encoded_image)
        logger.info(
            f"Base64 image successfully decoded for model: {model_key}")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid base64 string in 'encoded_image'")

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Image processing failed: {e}")

    try:
        model = MODELS[model_key]
        class_names = CLASS_NAMES.get(model_key, [])
        preds = model.predict(img_array)
        print(preds)

        preds = model.predict(img_array)
        logger.info(f"Raw sigmoid output: {preds[0][0]:.4f}")

        # Determine class based on sigmoid threshold 0.5
        if preds[0][0] >= 0.5:
            predicted_class = class_names[1]  # e.g. "car"
            confidence = float(preds[0][0])
        else:
            predicted_class = class_names[0]  # e.g. "bike"
            confidence = 1 - float(preds[0][0])

        logger.info(
            f"Prediction done: {predicted_class} with confidence {confidence:.4f}")

        logger.info(
            f"Prediction done: {predicted_class} with confidence {confidence:.4f}")
        return JSONResponse(content={
            "model_used": model_key,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail="Prediction failed due to server error.")
