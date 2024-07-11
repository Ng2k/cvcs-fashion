"""
La classe utilizza un modello di segmentation vestiti pre-addestrato.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from PIL import Image
import torch

from src.clothes_segmentation.interfaces.interface_segmentation_model import ISegmentationModel

class ClothesSegmentationController():
    """
    Classe usata per rappresentare un modello SegformerB2Clothes.
    """

    def __init__(self, model: ISegmentationModel):
        """
        Inizializza un nuovo oggetto SingleShotDetector.
        """
        self._segmentation_model: ISegmentationModel = model

    def apply_segmentation(self, image: Image) -> torch.Tensor:
        """Applica la segmentazione all'immagine.

        Args:
        -------
            image (Image): immagine da processare

        Returns:
        -------
            torch.Tensor: immagine segmentata
        """
        return self._segmentation_model.apply_segmentation(image)
