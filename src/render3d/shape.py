from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_SHAPE_MODEL = "facebook/dinov2-base"


class ShapeEncoder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vectors = self.model(**inputs).pooler_output
        vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        return vectors.cpu().numpy().astype(np.float32)


def model_vector(encoder: ShapeEncoder, paths: list[Path]) -> np.ndarray:
    images = [Image.open(path).convert("RGB") for path in paths]
    mean = encoder.encode(images).mean(axis=0)
    return mean / np.linalg.norm(mean)


def cache_name(model_name: str) -> str:
    return "shape-" + model_name.replace("/", "--") + ".npy"
