"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from abc import ABC, abstractmethod
from PIL import Image
import torch

class ISegmentationModel(ABC):
    """
    Interfaccia per i modelli di segmentazione.
    """

    @abstractmethod
    def apply_segmentation(self, image: Image) -> torch.Tensor:
        """Applica la segmentazione all'immagine.

        Args:
        -------
            image (Image): immagine da processare

        Returns:
        -------
            torch.Tensor: immagine segmentata
        """
