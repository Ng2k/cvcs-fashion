"""
Questo modulo definisce una classe base astratta per i modelli di segmentazione.

Interfaccia che specifica i metodi che devono essere implementati da qualsiasi modello concreto.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from abc import ABC, abstractmethod
from PIL import Image

class SegmentationModel(ABC):
    """
    Classe base astratta per i modelli di segmentazione.

    Metodi
    -------
        load_image(self, image_url: str) -> dict
            Metodo astratto
            Carica l'immagine da un URL
        find_best_bboxes(self, image_tensor: torch.Tensor) -> list
            Metodo astratto
            Trova le migliori bounding box per l'immagine.
    """

    @abstractmethod
    def apply_segmentation(self, image: Image) -> Image:
        """Applica la segmentazione all'immagine.

        Args:
        -------
            image (Image): immagine da processare

        Returns:
        -------
            Image: immagine segmentata
        """
