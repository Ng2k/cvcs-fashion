"""
Classe responsabile per l'elaborazione delle immagini.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from PIL import Image

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
    def crop_image(input_image: Image, crop_coordinates: tuple) -> Image:
        """
        Ritaglia l'immagine data le coordinate di un tensore PyTorch.

        Parametri
        ----------
            input_image : PIL.Image
                L'immagine da ritagliare.
            crop_coordinates : tuple
                Tupla di coordinate per il crop.
                (left, top, right, bottom)

        Ritorna
        -------
            PIL.Image
                L'immagine ritagliata.
        """
        return input_image.crop(crop_coordinates)
