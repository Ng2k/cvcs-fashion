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
    def crop_image_from_bbox(input_image: np.ndarray, bbox: tuple) -> np.ndarray:
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

        image_normalized: np.ndarray = (input_image / 2 + 0.5) * 255

        # Crop
        image_cropped: np.ndarray = image_normalized[y:y+h, x:x+w]

        return image_cropped

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
    def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
        """
        Ridimensiona l'immagine mantenendo l'aspect ratio.

        Parametri
        ----------
            image : np.ndarray
                L'immagine da ridimensionare.
            size : tupla
                La dimensione per il resize

        Ritorna
        -------
            np.ndarray
                L'immagine ridimensionata.
        """
        # Resize immagine usanto INTER_AREA per meno rumore possibile
        resized_image = cv2.resize(image, size, interpolation = cv2.INTER_AREA)

        return resized_image
      
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """
        Applica il denoising all'immagine utilizzando il Bilater Filter.

        Parametri
        ----------
            image : np.ndarray
                Immagine input da filtrare dal rumore.

        Ritorna
        -------
            np.ndarray
                Immagine filtrata.
        """

        denoised_image = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised_image
