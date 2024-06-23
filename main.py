"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

from src.similarity_calculator.cosine_similarity_function import CosineSimilarityFunction
from src.similarity_calculator.similarity_calculator import SimilarityFunction

from src.features_extractor.open_clip.model import OpenClipModel
from src.features_extractor.feature_extractor import FeatureExtractor

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor
from src.mask_processor import MaskProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation

def load_image(image_url: str) -> np.ndarray:
    return cv2.imread(image_url)

def ssd_detection(image_url: str) -> np.ndarray:
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> torch.Tensor:
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    return segmentation_model.apply_segmentation(image)

def features_extraction(masks: dict) -> torch.Tensor:
    feature_extractor = FeatureExtractor(OpenClipModel())
    return feature_extractor.extract_masks_features(masks)

def calculate_similarity(features: list[torch.Tensor]) -> torch.Tensor:
    similarity_function = SimilarityFunction(CosineSimilarityFunction())
    return similarity_function.compute_similarity(features)

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato
    """
    img_path = "./static/image_test_3"
    img_ext = ".jpg"

    #salvataggio dimensione immagine di input
    input_image =  load_image(img_path + img_ext)
    #input_shape = input_image.shape[:2]

    # Denoise dell'immagine
    denoise_image = ImageProcessor.denoise_image(input_image)

    # Ridimensiona l'immagine
    resized_image = ImageProcessor.resize_image(denoise_image, (300, 300))
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
    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path+"_segmented"+img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Applica le maschere
    masks = MaskProcessor.compute_masks(detected_image, segmented_image)

    features = features_extraction(masks)
    similarity_matrix = calculate_similarity(features)
    mean_similarity = similarity_matrix.mean(dim=0)
    lowest_similarity, index = mean_similarity.squeeze(0).min(dim=0)

    print(mean_similarity)
    print(lowest_similarity)
    print(index)

if __name__ == "__main__":
    main()
