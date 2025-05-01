from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/model_1.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_base64_image(data: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(data)
        image = np.array(Image.open(BytesIO(image_data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

class ImageData(BaseModel):
    base64_image: str

@app.post("/predict")
async def predict(image_data: ImageData):
    image = read_base64_image(image_data.base64_image)
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
