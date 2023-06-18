from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import base64
import uvicorn
import cv2
import numpy as np
import os
from keras.models import load_model
from model import train_model
from PIL import Image

from io import BytesIO

app = FastAPI()
# Use Jinja2Templates for rendering templates
templates = Jinja2Templates(directory="static")


# Serve static files directly from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")


def center_digit(image):
    # Load image as grayscale and obtain bounding box coordinates
    height, width = image.shape
    x, y, w, h = cv2.boundingRect(image)

    # Create new blank image
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Calculate the new coordinates for the ROI in the mask
    x_mask = width // 2 - w // 2
    y_mask = height // 2 - h // 2

    # Copy the ROI to the new coordinates in the mask
    mask[y_mask : y_mask + h, x_mask : x_mask + w] = image[y : y + h, x : x + w]

    return mask


# send the front end on landing page request 
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


dst_path = "data"
dst_extension = ".png"


class ImageData(BaseModel):
    image: str
    label: str


@app.post("/train")
def train_image(image_data: ImageData):
    image = image_data.image
    label = image_data.label

    dest_path = f"static/{dst_path}/{label}"

    # Get a list of all files in the folder
    files = os.listdir(dest_path)

    # Filter and extract only the image files with the extension
    image_files = [file for file in files if file.endswith(dst_extension)]

    # Extract the file names without extension and convert to integers
    image_names = [int(os.path.splitext(file)[0]) for file in image_files]

    # Find the largest integer name
    largest_name = max(image_names) if image_names else 0
    largest_name += 1

    image = image.replace("data:image/png;base64,", "")
    image_bytes = base64.urlsafe_b64decode(image)

    # Convert image bytes to numpy array
    image_array = np.array(Image.open(BytesIO(image_bytes)).convert("L"))

    # Center the digit in the image
    centered_image = center_digit(image_array)

    # Resize the centered image to 20x20
    img = Image.fromarray(centered_image)
    img = img.resize((20, 20))

    # save the file to destination folder 
    with open(f"{dest_path}/{largest_name}{dst_extension}", "wb") as file:
        img.save(file, "PNG")

    # Train the model
    train_model("model.h5")
    return JSONResponse({"train_complete": 1})


@app.post("/predict")
def predict_image(image_data: ImageData):
    image = image_data.image
    # Remove the data URL prefix before saving the image
    image = image.replace("data:image/png;base64,", "")

    image_bytes = base64.urlsafe_b64decode(image)

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode numpy array into image
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Center the digit in the image
    centered_image = center_digit(img_np)

    # Resize the centered image to 20x20
    resized_image = cv2.resize(centered_image, (20, 20))

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Reshape the image to have a single sample in the batch
    preprocessed_image = normalized_image.reshape((1, 20, 20, 1))

    loaded_model = load_model("model.h5")

    # predict image 
    prediction = loaded_model.predict(preprocessed_image)

    predicted_class = int(np.argmax(prediction))

    confidence = prediction[0][predicted_class] * 100

    if confidence > 50:
        confidence = "{:.2f} %".format(float(confidence))

    else:
        confidence = "digit not recognized"
        predicted_class = -1

    print(predicted_class, confidence)
    return JSONResponse(
        {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }
    )


if __name__ == "__main__":
    train_model("model.h5")
    uvicorn.run(app, host="127.0.0.1", port=4000)
