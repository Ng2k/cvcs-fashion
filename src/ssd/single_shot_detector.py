"""
La classe utilizza un modello SSD pre-addestrato.
Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from src.image_processor import ImageProcessor
from src.ssd.ssd_model import SSDModel

class SingleShotDetector():
    """
    Classe usata per rappresentare un Single Shot Detector (SSD) model.

    Attributi
    ----------
        _CONFIDENCE : float
            Attributo privato
            Threshold di confidenza per la detection degli oggetti.
            Gli oggetti con un punteggio di confidenza inferiore a questa soglia vengono ignorati.
        _ssd_model : SSDModel
            Modello SSD.
    """

    _CONFIDENCE: float = 0.40

    def __init__(self, model: SSDModel):
        """
        Inizializza un nuovo oggetto SingleShotDetector.
        """
        self._ssd_model: SSDModel = model

    def _retrieve_image_cropped(self, image_numpy: np.ndarray, bboxes: list) -> Image:
        """Ritorna l'immagine ritagliata in base alla detection.

        Args:
        -------
            image_numpy (np.ndarray): immagine caricata come array numpy
            bboxes (list): bounding box

        Returns:
        -------
            Image: immagine ritagliata
        """
        for image_result in bboxes:
            for _, bbox in enumerate(image_result[0]):
                return ImageProcessor.crop_image_from_bbox(image_numpy, bbox)

    def detect_person_in_image(self, image_url: str) -> None:
        """Funzione per la detection

        Args:
        -------
            image_url (str): url dell'immagine
        """
        image_loaded = self._ssd_model.load_image(image_url)
        image_numpy, image_tensor = image_loaded["image_numpy"], image_loaded["image_tensor"]

        bboxes = self._ssd_model.find_best_bboxes(image_tensor)

        image_cropped = self._retrieve_image_cropped(image_numpy, bboxes)

        plt.imshow(image_cropped)
