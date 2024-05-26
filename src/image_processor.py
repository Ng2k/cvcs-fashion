"""
Classe responsabile per l'elaborazione delle immagini.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from PIL import Image
import numpy as np

class ImageProcessor:
    """
    Classe responsabile per l'elaborazione delle immagini.

    Questa classe fornisce un metodo per ritagliare un'immagine 
    data le coordinate di un tensore PyTorch.

    Attributi
    ----------
        image : PIL.Image
            L'immagine da elaborare.
    """

    @staticmethod
    def crop_image_from_bbox(image: np.ndarray, bbox: tuple) -> Image:
        """
        Ritaglia l'immagine, tensore pytorch, data delle coordinate.

        Parametri
        ----------
            input_image : np.ndarray
                L'immagine da ritagliare.
            bbox : tuple
                Tupla di coordinate della bounding box.

        Ritorna
        -------
            torch.Tensor
                L'immagine ritagliata.
        """

        left, bot, right, top = bbox
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        image = image / 2 + 0.5
        image = Image.fromarray(np.uint8(image * 255))
        image = image.crop((x, y, x + w, y + h))

        return image

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
    def resize_image(input_image: Image, size: tuple) -> Image:
        """
        Ridimensiona l'immagine data le dimensioni.

        Parametri
        ----------
            input_image : PIL.Image
                L'immagine da ridimensionare.
            size : tuple
                Dimensioni della nuova immagine.
                (width, height)

        Ritorna
        -------
            PIL.Image
                L'immagine ridimensionata.
        """
        return input_image.resize(size)
