"""
La classe utilizza un modello SSD pre-addestrato.
Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from src.image_processor import ImageProcessor
from src.ssd.ssd_model import SSDModel

class SingleShotDetector():
    """
    Classe usata per rappresentare un Single Shot Detector (SSD) model.

    Usa un modello SSD pre-addestrato da NVIDIA's Deep Learning Examples.
    Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

    Attributi
    ----------
        _CONFIDENCE : float
            Attributo privato
            Threshold di confidenza per la detection degli oggetti.
            Gli oggetti con un punteggio di confidenza inferiore a questa soglia vengono ignorati.
        _image : np.ndarray
            Attributo privato
            L'immagine da elaborare, rappresentata come un array NumPy.
        _image_tensor : torch.Tensor
            Attributo privato
            L'immagine da elaborare, rappresentata come un tensore PyTorch.
        ssd_model : any
                Modello SSD.
        utils : any
            Utils modello SSD.
    """

    _CONFIDENCE: float = 0.40

    _image: np.ndarray
    _image_tensor: torch.Tensor

    def __init__(self, model: SSDModel):
        """
        Inizializza un nuovo oggetto SingleShotDetector.
        """
        self.ssd_model = model.load_model()
        self.utils = model.load_utils()
        self._image = None
        self._image_tensor = None

    def _load_image(self, image_url: str) -> None:
        """ Funzione per il caricamento dell'immagine.

        Args:
        -------
            image_url (str): url dell'immagine
        """
        self._image = self.utils.prepare_input(image_url)
        self._image_tensor = self.utils.prepare_tensor([self._image])

    def _find_best_bboxes(self, image_tensor: torch.Tensor) -> list:
        """ Trova le migliori bounding box per l'immagine.

        Args:
        -------
            image (torch.Tensor): immagine da processare

        Returns:
        -------
            detection (list): lista delle detection
        """
        with torch.no_grad():
            detections_batch = self.ssd_model(image_tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [
            self.utils.pick_best(results, self._CONFIDENCE) for results in results_per_input
        ]

        return best_results_per_input

    def _retrieve_image_cropped(self) -> Image:
        """Ritorna l'immagine ritagliata in base alla detection.

        Returns:
        -------
            Image: immagine ritagliata
        """
        for image_result in self._find_best_bboxes(self._image_tensor):
            for _, bbox in enumerate(image_result[0]):
                return ImageProcessor.crop_image_from_bbox(self._image, bbox)

    def detect_person_in_image(self, image_url: str) -> None:
        """Funzione per la detection

        Args:
        -------
            image_url (str): url dell'immagine
        """
        self._load_image(image_url)
        plt.imshow(self._retrieve_image_cropped())