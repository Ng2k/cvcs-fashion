"""
Classe responsabile per l'elaborazione delle immagini.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from PIL import Image
import numpy as np
import cv2

class ImageProcessor:
    """
    Classe responsabile per l'elaborazione delle immagini.

    Questa classe fornisce un metodo per ritagliare un'immagine 
    data le coordinate di un tensore PyTorch.
    """

    @staticmethod
    def crop_image_from_bbox(image: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Ritaglia l'immagine data delle coordinate della bounding box.

        Parametri
        ----------
        image : np.ndarray
            L'immagine da ritagliare.
        bbox : tuple
            Tupla di coordinate della bounding box.

        Ritorna
        -------
        np.ndarray
            L'immagine ritagliata.
        """
        left, bot, right, top = bbox
        x, y, w, h = [int(val * 300) for val in [left, bot, right - left, top - bot]]

        # Controllo se l'immagine è [0, 255]
        image = np.uint8(image * 255)

        # Crop con OpenCV
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image

    @staticmethod
    def tilt_image(input_image: Image) -> Image:
        """
        Inclina l'immagine data le coordinate di un tensore PyTorch.

        Parametri
        ----------
            input_image : PIL.Image
                L'immagine da inclinare.

        Ritorna
        -------
            PIL.Image
                L'immagine inclinata.
        """
        return input_image

    @staticmethod
    def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Ridimensiona l'immagine mantenendo l'aspect ratio.

        Parametri
        ----------
            image : np.ndarray
                L'immagine da ridimensionare.
            max_size : int
                La dimensione massima del lato più lungo dell'immagine ridimensionata.

        Ritorna
        -------
            np.ndarray
                L'immagine ridimensionata.
        """
        height, width = image.shape[:2]

        # Determinazione scale factor mantenendo aspect ratio
        if height > width:
            scale_factor = max_size / float(height)
        else:
            scale_factor = max_size / float(width)

        new_size = (int(width * scale_factor), int(height * scale_factor))

        # Resize immagine usanto INTER_AREA per meno rumore possibile
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        return resized_image
