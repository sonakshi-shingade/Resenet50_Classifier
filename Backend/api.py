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

# -------------------------------
# ✅ Setup Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# ✅ FastAPI App Initialization
# -------------------------------
app = FastAPI(
    title="Image Classification API",
    description="This API allows predictions using pre-trained models for binary and multiclass classification.",
    version="1.0"
)

# -------------------------------
# ✅ Model Paths & Class Names
# -------------------------------
MODEL_PATHS = {
    "bike_car_model": "resnet50_car_bike.h5",
    "animal_model": "resnet50_animal.h5",
    "rice_model": "resnet50_rice.h5"
}

CLASS_NAMES = {
    "bike_car_model": ["bike", "car"],
    "animal_model": ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
                     "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"],
    "rice_model": ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
}

# -------------------------------
# ✅ Load Models Once at Startup
# -------------------------------
MODELS = {}
for model_name, file_path in MODEL_PATHS.items():
    try:
        MODELS[model_name] = load_model(file_path)
        logger.info(f"✅ Loaded model: {model_name} from {file_path}")
    except Exception as e:
        logger.error(f"⚠️ Could not load model '{model_name}' from '{file_path}': {e}")

# -------------------------------
# ✅ Request Schema
# -------------------------------
class PredictRequest(BaseModel):
    model: str
    classifier: str  # "binary" or "multiclass"
    encoded_image: str


# -------------------------------
# ✅ Image Preprocessing
# -------------------------------
def preprocess_image(image_bytes):
    """
    Preprocesses the input image bytes to match model input format.
    
    Args:
        image_bytes (bytes): Raw image in bytes.
    
    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        logger.info("🔄 Starting image preprocessing...")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logger.info("✅ Image preprocessing completed.")
        return img_array
    except Exception as e:
        logger.error(f"❌ Error in preprocessing image: {e}")
        raise


# -------------------------------
# ✅ Prediction Endpoint
# -------------------------------
@app.post("/predict/")
async def predict(request: PredictRequest):
    """
    Predicts the class of an input image using a specified model and classifier type.

    Args:
        request (PredictRequest): JSON body containing model name, classifier type, and base64 image.

    Returns:
        JSONResponse: Contains model used, predicted class, and confidence score.
    """
    model_key = request.model
    classifier = request.classifier.lower()
    encoded_image = request.encoded_image

    logger.info(f"📦 Incoming request → Model: '{model_key}' | Classifier: '{classifier}'")

    # Validate model name
    if model_key not in MODELS:
        logger.warning(f"❌ Model not found: {model_key}")
        raise HTTPException(status_code=400, detail=f"Model '{model_key}' not found.")

    # Validate classifier
    if classifier not in ["binary", "multiclass"]:
        logger.warning(f"❌ Invalid classifier: {classifier}")
        raise HTTPException(status_code=400, detail="Classifier must be 'binary' or 'multiclass'.")

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(encoded_image)
        logger.info("🧬 Image decoded from base64.")
    except Exception as e:
        logger.error(f"❌ Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 string.")

    # Preprocess image
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    # Prediction logic
    try:
        model = MODELS[model_key]
        class_names = CLASS_NAMES.get(model_key, [])

        preds = model.predict(img_array)
        logger.info(f"📈 Model prediction raw output: {preds}")

        # Binary classification
        if classifier == "binary":
            pred_value = preds[0][0]
            predicted_class = class_names[1] if pred_value >= 0.5 else class_names[0]
            confidence = float(pred_value) if pred_value >= 0.5 else 1 - float(pred_value)

        # Multiclass classification
        else:
            predicted_index = int(np.argmax(preds[0]))
            predicted_class = class_names[predicted_index]
            confidence = float(preds[0][predicted_index])

        logger.info(f"✅ Prediction successful → Class: {predicted_class}, Confidence: {confidence:.4f}")

        return JSONResponse(content={
            "model_used": model_key,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to server error.")
