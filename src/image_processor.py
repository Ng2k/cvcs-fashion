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
    def resize_image(input_image: Image, max_size: int) -> Image:
        """
        Ridimensiona l'immagine data le dimensioni.
        Mantiene aspect ratio e qualità immagine al meglio possibile

        Parametri
        ----------
            input_image : PIL.Image
                L'immagine da ridimensionare.
            max_size : int
                Dimensione massima dell'immagine

        Ritorna
        -------
            PIL.Image
                L'immagine ridimensionata.
        """
        original_width, original_height = input_image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        return input_image.resize((new_width, new_height))
