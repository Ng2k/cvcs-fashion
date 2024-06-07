"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor
from src.mask_processor import MaskProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation

def load_image(image_url: str) -> np.ndarray:
    """Carica un'immagine da un file e la converte in un array NumPy.
    
    Parameters
    ----------
        image_url : str
            Percorso dell'immagine da caricare.
    Returns
    -------
        np.ndarray
            L'immagine caricata.
    """
    return cv2.imread(image_url)

def image_resize(image: np.ndarray, max_size: int) -> np.ndarray:
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

def segmentation(image : Image) -> torch.Tensor:
    """
    Applica la segmentazione dei vestiti all'immagine utilizzando il modello di segmentazione.

    Parametri
    ----------
        image : Image
            Immagine su cui applicare la segmentazione.

    Ritorna
    -------
        torch.Tensor
            L'immagine con la segmentazione dei vestiti.
    """
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    segmented_image = segmentation_model.apply_segmentation(image)
    return segmented_image

def apply_masks(input_image: np.ndarray, segmented_image: np.ndarray) -> dict:
    """
    Applica le maschere di segmentazione all'immagine originale.

    Parametri
    ----------
        input_image : np.ndarray
            Immagine originale
        segmented_image : np.ndarray
            Immagine segmentata

    Ritorna
    -------
        dict
            Le immagini con le maschere applicate.
    """
    return MaskProcessor.compute_masks(input_image, segmented_image)

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato finale.
    """
    img_path = "./static/image_test_2"
    img_ext = ".jpg"

    #salvataggio dimensione immagine di input
    input_image =  load_image(img_path + img_ext)
    # input_shape = input_image.shape[:2]

    # Denoise dell'immagine
    denoise_image = ImageProcessor.denoise_image(input_image)

    # Ridimensiona l'immagine
    size = (300, 300)
    resized_image = image_resize(denoise_image, size)
    cv2.imwrite(img_path + "_resized" + img_ext, resized_image)

    # Esegue la rilevazione della persona
    detected_image = ssd_detection(img_path + "_resized" + img_ext)
    detected_image_pil = Image.fromarray(detected_image.astype(np.uint8))
    detected_image_pil.save(img_path + "_crop" + img_ext)

    # Ritorno alle dimensioni originali
    #resized_back_image = cv2.resize(
    #    detected_image.astype(np.uint8),
    #    (input_shape[1], input_shape[0])
    #)

    # Applica la segmentazione dei vestiti
    segmented_image = segmentation(detected_image_pil)

    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path + "_segmented" + img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Applica le maschere
    masks = apply_masks(detected_image, segmented_image)

    plt.imshow(masks[4])
    plt.show()

if __name__ == "__main__":
    main()
