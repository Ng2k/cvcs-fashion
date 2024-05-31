"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation

def image_resize(image_path: str, max_size: int) -> np.ndarray:
    """
    Ridimensiona un'immagine al massimo valore specificato mantenendo l'aspect ratio.

    Parametri
    ----------
        image_path : str
            Percorso dell'immagine da ridimensionare.
        max_size : int
            Dimensione massima del lato più lungo dell'immagine ridimensionata.

    Ritorna
    -------
        np.ndarray
            L'immagine ridimensionata.
    """
    image = np.array(Image.open(image_path))
    resized_image = ImageProcessor.resize_image(image, max_size)
    return resized_image

def ssd_detection(image_url: str) -> np.ndarray:
    """
    Esegue la rilevazione della persona nell'immagine utilizzando il modello SSD.

    Parametri
    ----------
        image : np.ndarray
            Immagine su cui eseguire la rilevazione.

    Ritorna
    -------
        np.ndarray
            L'immagine con la persona individuata.
    """
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> Image:
    """
    Applica la segmentazione dei vestiti all'immagine utilizzando il modello di segmentazione.

    Parametri
    ----------
        image : Image
            Immagine su cui applicare la segmentazione.

    Ritorna
    -------
        Image
            L'immagine con la segmentazione dei vestiti.
    """
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    segmented_image = segmentation_model.apply_segmentation(image)
    return segmented_image

def main():
    img_path = "OIP2"
    img_ext = ".jpg"
    # Ridimensiona l'immagine
    size = (300, 300)
    resized_image = image_resize("./" + img_path + img_ext, size)

    resized_image_PIL = Image.fromarray(resized_image.astype(np.uint8))
    resized_image_PIL.save("./" + img_path + "_resized" + img_ext)

    # Esegue la rilevazione della persona
    detected_image = ssd_detection("./" + img_path + "_resized" + img_ext)
    detected_image_PIL = Image.fromarray(detected_image.astype(np.uint8))
    detected_image_PIL.save("./" + img_path + "_crop" + img_ext)

    denoise_image = ImageProcessor.denoise_image(detected_image.astype(np.uint8))
    denoise_image_PIL = Image.fromarray(denoise_image.astype(np.uint8))

    # Applica la segmentazione dei vestiti
    segmented_image = segmentation(denoise_image_PIL)

    # Mostra l'immagine segmentata
    plt.imshow(segmented_image)


if __name__ == "__main__":
    main()
