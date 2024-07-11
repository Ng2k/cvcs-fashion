"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import json
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
import os

from src.mask_generator.classes.mask_generator import MaskGenerator
from src.outfit_component_selector.classes.outfit_component_selector_base import OutfitComponentSelector

from src.similarity_calculator.classes.cosine_similarity_function import CosineSimilarityFunction
from src.similarity_calculator.similarity_controller import SimilarityController

from src.single_shot_detector.classes.nvidia_single_shot_detector import NvidiaSingleShotDetector
from src.single_shot_detector.single_shot_detector import SingleShotDetector

from src.clothes_segmentation.classes.segformer_b2_clothes import SegformerB2Clothes
from src.clothes_segmentation.clothes_segmentation_controller import ClothesSegmentationController

from src.feature_extractor.feature_extractor import FeatureExtractor
from src.feature_extractor.classes.open_clip import OpenClip

from src.label_mapper.classes.label_mapper import LabelMapper
from src.image_processor import ImageProcessor
from src.utils import Utils

def load_image(image_url: str) -> np.ndarray:
    return cv2.imread(image_url)

def ssd_detection(image_url: str) -> np.ndarray:
    ssd_model = SingleShotDetector(NvidiaSingleShotDetector())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> torch.Tensor:
    segmentation_model = ClothesSegmentationController(SegformerB2Clothes())
    return segmentation_model.apply_segmentation(image)

def delete_images(img_path: str, img_ext: str) -> None:
    files_to_delete = [
        img_path + "_resized" + img_ext,
        img_path + "_crop" + img_ext,
        img_path + "_segmented" + img_ext,
    ]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"The file {file_path} does not exist")

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato
    """
    img_path = "./static/image_test"
    img_ext = ".jpg"

    #salvataggio dimensione immagine di input
    input_image =  load_image(img_path + img_ext)

    # Denoise dell'immagine
    denoise_image = ImageProcessor.denoise_image(input_image)

    # Ridimensiona l'immagine
    resized_image = ImageProcessor.resize_image(denoise_image, (300, 300))
    cv2.imwrite(img_path + "_resized" + img_ext, resized_image)

    # Esegue la rilevazione della persona
    detected_image = ssd_detection(img_path + "_resized" + img_ext)
    detected_image_pil = Image.fromarray(detected_image.astype(np.uint8))
    detected_image_pil.save(img_path + "_crop" + img_ext)

    # Applica la segmentazione dei vestiti
    segmented_image = segmentation(detected_image_pil)
    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path+"_segmented"+img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Rimozione immagini temporanee
    delete_images(img_path, img_ext)

    # Applica le maschere    
    mask_generator = MaskGenerator()
    mask_list = mask_generator.generate_masks(detected_image, segmented_image)

    outfit_component_selector = OutfitComponentSelector(
        mask_list,
        FeatureExtractor(OpenClip()),
        SimilarityController(CosineSimilarityFunction()),
        LabelMapper(OpenClip())
    )

    polyvore_image = outfit_component_selector.find_best_fit(Utils.get_prompt_list())
    plt.imshow(polyvore_image)
    plt.show()

if __name__ == "__main__":
    main()
