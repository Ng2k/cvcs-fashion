"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from abc import ABC, abstractmethod
import torch

from src.single_shot_detector.types.image_type import ImageSingleShotDetector

class ISingleShotDetector(ABC):
    """
    Interfaccia per i modelli SSD
    """

    @abstractmethod
    def load_image(self, image_url: str) -> ImageSingleShotDetector:
        """Carica l'immagine in 2 formati: numpy e tensore PyTorch.

        Args:
        -------
            image_url (str): percorso immagine

        Returns:
        -------
            ImageSingleShotDetector: dizionario con l'immagine caricata come array numpy e tensore PyTorch
        """

    @abstractmethod
    def find_best_bboxes(self, image_tensor: torch.Tensor) -> list:
        """ Trova le migliori bounding box per l'immagine.

        Args:
        -------
            image (torch.Tensor): immagine da processare

        Returns:
        -------
            detection (list): lista delle detection
        """
