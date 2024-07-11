"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class ImageSingleShotDetector():
    """Tipo dato per immagini caricate con il modello di SingleShotDetector
    """
    image_numpy: np.ndarray
    image_tensor: torch.Tensor