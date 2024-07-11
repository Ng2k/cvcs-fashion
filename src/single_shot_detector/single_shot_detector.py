"""
La classe utilizza un modello SSD pre-addestrato.
Fornisce metodi per caricare un'immagine, elaborarla e disegnare dei bounding box.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

import numpy as np

from src.image_processor import ImageProcessor
from src.single_shot_detector.interfaces.interface_single_shot_detector import ISingleShotDetector

class SingleShotDetector():
    """
    Classe usata per rappresentare un Single Shot Detector (SSD) model.
    """

    def __init__(self, model: ISingleShotDetector):
        """
        Inizializza un nuovo oggetto SingleShotDetector.
        """
        self._ssd_model: ISingleShotDetector = model

    def _retrieve_image_cropped(self, image_numpy: np.ndarray, bboxes: list) -> np.ndarray:
        """Ritorna l'immagine ritagliata in base alla detection.

        Args:
        -------
            image_numpy (np.ndarray): immagine caricata come array numpy
            bboxes (list): bounding box

        Returns:
        -------
            np.ndarray: immagine ritagliata
        """
        return ImageProcessor.crop_image_from_bbox(image_numpy, bboxes)

    def detect_person_in_image(self, image_url: str) -> np.ndarray:
        """Funzione per la detection

        Args:
        -------
            image_url (str): url dell'immagine

        Returns:
        -------
            np.ndarray: immagine con la persona individuata
        """
        image_loaded = self._ssd_model.load_image(image_url)
        image_numpy, image_tensor = image_loaded["image_numpy"], image_loaded["image_tensor"]
        bboxes = self._ssd_model.find_best_bboxes(image_tensor)
        return self._retrieve_image_cropped(image_numpy, bboxes)
