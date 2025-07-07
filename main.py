from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import uuid
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List

from app.services.segmentator_factory import get_segmentator
from app.utils import decode_base64_image, encode_image_to_base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Label(BaseModel):
    id: int
    name: str
    prompt: str

class GenerateImageVariants(BaseModel):
    number_of_images: int
    image: str
    aspect_ratio: str
    prompt: str
    negative_prompt: str
    annotation_model: str
    strength: float = 1
    labels: List[Label]

@app.post("/generate_images/image_variants/")
async def generate_image_variants(body: GenerateImageVariants):
    image = decode_base64_image(body.image)
    np_image = np.array(image)
    h_img, w_img, _ = np_image.shape
    segmentator = get_segmentator(body.annotation_model, body.labels)
    results = segmentator.predict(np_image)
    annotations = segmentator.build_annotations(results, w_img, h_img)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = encode_image_to_base64(buffer.getvalue())
    return {
        "image": "data:image/jpeg;base64," + img_str,
        "annotations": annotations
    }
