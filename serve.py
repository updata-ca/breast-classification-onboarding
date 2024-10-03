from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import base64
import numpy as np
from PIL import Image
import cv2

# Define a Pydantic model for the request body
class Image(BaseModel):
    serialized_img: str

# Create a FastAPI instance
app = FastAPI()

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)


# POST endpoint that accepts JSON data
@app.post("/infer/")
async def create_item(item: Image):
    
    ort_sess = ort.InferenceSession('output/model.onnx')

    # image_data = base64.b64decode(item.serialized_img)
    # image = Image.open(BytesIO(image_data)).convert('L') 
    
    # image = image.resize((224, 224))
    # image_array = np.array(image).astype(np.float32) / 255.0
    
    # # image_array = np.frombuffer(image_data, dtype=np.uint8)
    # image_array = image_array.reshape((224, 224, 1))

    img_base64 = item.serialized_img
    decoded_img_bytes = base64.b64decode(img_base64)
    nparr = np.frombuffer(decoded_img_bytes, np.uint8)
    img_decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_decoded = img_decoded.reshape(1, 224, 224, 1).astype(np.float32) / 255.0    
    outputs = ort_sess.run(None, {'x': img_decoded})

    predicted = int(softmax(outputs[0][0]).argmax(0))
    
    return {
        "predicted": predicted,
    }

# To run the server, use: uvicorn filename:app --reload