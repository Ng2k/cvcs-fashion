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
from src.utils import Utils

def load_image(image_url: str) -> np.ndarray:
    return cv2.imread(image_url)

def ssd_detection(image_url: str) -> np.ndarray:
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> torch.Tensor:
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    return segmentation_model.apply_segmentation(image)

def mapping_label_to_mask(masks: dict) -> list:
    feature_extractor = FeatureExtractor(OpenClipModel())
    with open("./prompts_no_desc.json", "r") as prompts_file:
        prompts = json.load(prompts_file).get('prompts')

    map_label_mask_list = []
    for mask in masks.values():
        img_tensor = feature_extractor.model.load_and_process_image(mask)
        inferences = feature_extractor.model.run_inference(img_tensor, prompts)
        _, index_label = inferences.squeeze(0).max(dim=0)
        map_label_mask_list.append({
            "idx_label": index_label.item(),
            "label": prompts[index_label.item()],
            "mask": mask
        })
    
    return map_label_mask_list

def features_extraction(masks: dict) -> list:
    feature_extractor = FeatureExtractor(OpenClipModel())
    return feature_extractor.extract_masks_features(masks)

def calculate_similarity(features) -> torch.Tensor:
    similarity_function = SimilarityFunction(CosineSimilarityFunction())
    return similarity_function.compute_similarity(features)

def find_index_mask_to_replace(masks: dict) -> int:
    features = features_extraction(masks)
    similarity_matrix = calculate_similarity(features)
    mean_similarity = similarity_matrix.mean(dim=0)
    _, index_lowest_similarity = mean_similarity.squeeze(0).min(dim=0)
    return index_lowest_similarity

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato
    """
    img_path = "./static/image"
    img_ext = ".jpeg"

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

    # Applica la segmentazione dei vestiti
    segmented_image = segmentation(detected_image_pil)
    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path+"_segmented"+img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Applica le maschere
    masks = MaskProcessor.compute_masks(detected_image, segmented_image)
    masks_features = features_extraction(masks)

    # inferenza maschere con classi open clip
    map_label_mask_list = mapping_label_to_mask(masks)

    # selezione maschera da rimpiazzare
    idx_mask_to_replace = find_index_mask_to_replace(masks)
    label_mask_to_replace = map_label_mask_list[idx_mask_to_replace.item()]["idx_label"]
    features_to_keep = [mask for i, mask in enumerate(masks_features) if i != idx_mask_to_replace]

    try:
        with open(f"./dataset/polyvore_feature_vectors/{label_mask_to_replace}.json", "r") as img_files:
            images = json.load(img_files).get('images')
    except FileNotFoundError:
        images = []

    mean_similarity_list = []
    for img in images:
        img_features = torch.tensor(img["features"]).to(Utils.get_device())
        similarity_matrix = calculate_similarity(features_to_keep + [img_features])
        mean_similarity = similarity_matrix.mean(dim=0)
        mean_similarity_list.append(mean_similarity[-1])

    mean_similarity_tensor = torch.tensor(mean_similarity_list)
    _, idx_best_similarity = mean_similarity_tensor.max(dim=0)

    #print(images[idx_best_similarity])
    plt.imshow(cv2.imread(images[idx_best_similarity]["path"]))
    plt.show()

if __name__ == "__main__":
    main()
