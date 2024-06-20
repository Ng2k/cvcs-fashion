"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from PIL import Image
import numpy as np

class ImageProcessorClip:
    """
    Classe ImageProcessor per il pre-processing delle immagini
    """
    def __init__(self, preprocess, device:str='cpu'):
        self.preprocess = preprocess
        self.device = device

    def load_and_process_image(self, image: np.ndarray) -> Image:
        """Carica e processa un'immagine

        Args:
        -------
            image (np.ndarray): percorso dell'immagine

        Returns:
        -------
            _type_: _description_
        """
        img = self.preprocess(Image.fromarray(image)).to(self.device)
        return img
