"""Feature extraction using ResNet50 model."""

import logging
from typing import List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from settings import settings


class FeatureExtractor:
    """Feature extraction using ResNet50 model for image embeddings."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.device = torch.device(settings.ml.device)
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self) -> nn.Module:
        """Load ResNet50 model without the final classification layer."""
        self.logger.info(f"Loading ResNet50 model with {settings.model.weights} weights")
        
        # Load pretrained ResNet50
        weights = getattr(models.ResNet50_Weights, settings.model.weights)
        model = models.resnet50(weights=weights)
        
        # Remove the final classification layer to get embeddings
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"ResNet50 model loaded successfully on {self.device}")
        return model

    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((settings.ml.img_size, settings.ml.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract ResNet50 features from a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Flatten and normalize embedding for cosine similarity
            embedding = features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return np.zeros(settings.ml.embedding_dim)

    def extract_features_from_pil(self, image: Image.Image) -> np.ndarray:
        """Extract ResNet50 features from a PIL Image."""
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Flatten and normalize embedding for cosine similarity
            embedding = features.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return np.zeros(settings.ml.embedding_dim)

    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from multiple images in batches."""
        all_features = []
        batch_size = settings.ml.batch_size
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting ResNet50 features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
                    # Create a blank tensor as fallback
                    batch_tensors.append(torch.zeros(3, settings.ml.img_size, settings.ml.img_size))
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.model(batch)
                
                # Flatten each embedding
                embeddings = features.cpu().numpy().reshape(len(batch_tensors), -1)
                # Normalize each embedding
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                embeddings = embeddings / norms
                all_features.append(embeddings)
        
        return np.vstack(all_features) if all_features else np.array([])
