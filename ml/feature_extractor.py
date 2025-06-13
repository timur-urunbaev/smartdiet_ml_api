"""Feature extraction using pre-trained models."""

import logging
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

from .config import Config

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Encapsulated feature extraction using pre-trained models."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> nn.Module:
        """Load and configure the pre-trained model."""
        self.logger.info(f"Loading {self.config.model_name} model...")

        # Support for different model architectures
        if self.config.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Identity() # type: ignore
        elif self.config.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier = nn.Identity() # type: ignore
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

        model = model.to(self.config.device).eval()
        self.logger.info(f"Model loaded successfully on {self.config.device}")
        return model

    def _get_transforms(self) -> T.Compose:
        """Get image preprocessing transforms."""
        return T.Compose([
            T.Resize((self.config.img_size, self.config.img_size)),
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
            # x = self.transform(img).unsqueeze(0).to(self.config.device)
            x = transformed.unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                features = self.model(x).cpu().numpy().flatten()

            return features
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return np.zeros(self.config.embedding_dim)

    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from multiple images in batches."""
        all_features = []

        for i in tqdm(range(0, len(image_paths), self.config.batch_size),
                     desc="Extracting features"):
            batch_paths = image_paths[i:i + self.config.batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(self.transform(img))
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
                    batch_images.append(torch.zeros(3, self.config.img_size, self.config.img_size))

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.config.device)

                with torch.no_grad():
                    batch_features = self.model(batch_tensor).cpu().numpy()

                all_features.append(batch_features)

        return np.vstack(all_features) if all_features else np.array([])
