"""Feature extraction using pre-trained models."""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

from settings import settings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Encapsulated feature extraction using pre-trained models."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> nn.Module:
        """Load and configure the pre-trained model."""
        self.logger.info(f"Loading {settings.ml.model_name} model...")

        # Support for different model architectures
        if settings.ml.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Identity() # type: ignore
        elif settings.ml.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier = nn.Identity() # type: ignore
        else:
            raise ValueError(f"Unsupported model: {settings.ml.model_name}")

        model = model.to(settings.ml.device).eval()
        self.logger.info(f"Model loaded successfully on {settings.ml.device}")
        return model

    def _get_transforms(self) -> T.Compose:
        """Get image preprocessing transforms."""
        return T.Compose([
            T.Resize((settings.ml.img_size, settings.ml.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image."""
        try:
            img = Image.open(image_path).convert("RGB")
            transformed = self.transform(img)
            if not isinstance(transformed, torch.Tensor):
                transformed = T.ToTensor()(transformed)
            # x = self.transform(img).unsqueeze(0).to(settings.ml.device)
            x = transformed.unsqueeze(0).to(settings.ml.device)

            with torch.no_grad():
                features = self.model(x).cpu().numpy().flatten()

            return features
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return np.zeros(settings.ml.embedding_dim)

    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from multiple images in batches."""
        all_features = []

        for i in tqdm(range(0, len(image_paths), settings.ml.batch_size),
                     desc="Extracting features"):
            batch_paths = image_paths[i:i + settings.ml.batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(self.transform(img))
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
                    batch_images.append(torch.zeros(3, settings.ml.img_size, settings.ml.img_size))

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(settings.ml.device)

                with torch.no_grad():
                    batch_features = self.model(batch_tensor).cpu().numpy()

                all_features.append(batch_features)

        return np.vstack(all_features) if all_features else np.array([])
