from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../paddy.h5")

CLASS_NAMES = ['bacterial_leaf_blight',
               'bacterial_leaf_streak',
               'bacterial_panicle_blight',
               'blast',
               'brown_spot',
               'dead_heart',
               'downy_mildew',
               'hispa',
               'normal',
               'tungro']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.options("/predict")
async def options_predict():
    return {"message": "Allow preflight requests"}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to 256x256 pixels
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    print("Image Read")
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    print("Predictions Done")

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        'class': predicted_class,
        'confidence': confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
