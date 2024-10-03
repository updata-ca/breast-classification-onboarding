import requests
import base64
import cv2
import numpy as np

# with open('example.jpg', 'rb') as f:
#     encoded_img = base64.b64encode(f.read()).decode('utf-8')

sample_img_path = 'example.jpg'
img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.uint8)
img = img.reshape(224, 224, 1)
_, img_encoded = cv2.imencode('.jpg', img)
img_bytes = img_encoded.tobytes()
img_base64 = base64.b64encode(img_bytes).decode('utf-8')

response = requests.post("http://localhost:8000/infer/", json={"serialized_img": img_base64})

print(response.json())