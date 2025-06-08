"""
Enhanced Image Retrieval System with Best Practices
Author: Senior ML Engineer
"""

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
import faiss
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
