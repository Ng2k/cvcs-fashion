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

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor
from src.mask_processor import MaskProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation

from src.open_fashion_clip.open_clip_model_manager import OpenClipModelManager
from src.open_fashion_clip.image_processor_clip import ImageProcessorClip
from src.open_fashion_clip.clip_inference import ClipInference
from src.open_fashion_clip.text_processor import TextProcessor

from src.utils import Utils

def load_image(image_url: str) -> np.ndarray:
    return cv2.imread(image_url)

def image_resize(image: np.ndarray, max_size: int) -> np.ndarray:
    resized_image = ImageProcessor.resize_image(image, max_size)
    return resized_image

def ssd_detection(image_url: str) -> np.ndarray:
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image_url)

def segmentation(image : Image) -> torch.Tensor:
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    segmented_image = segmentation_model.apply_segmentation(image)
    return segmented_image

def apply_masks(input_image: np.ndarray, segmented_image: np.ndarray) -> dict:
    return MaskProcessor.compute_masks(input_image, segmented_image)

def main():
    """
    Funzione principale dello script. Esegue i seguenti passaggi:

    Nota:   Questa funzione non restituisce nulla.
            Salva i risultati intermedi e finali su disco e mostra il risultato finale.
    """
    img_path = "./static/image_test"
    img_ext = ".jpg"

    #salvataggio dimensione immagine di input
    input_image =  load_image(img_path + img_ext)
    #input_shape = input_image.shape[:2]

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

    Image.fromarray(segmented_image.numpy().astype(np.uint8)).save(img_path+"_segmented"+img_ext)
    segmented_image = segmented_image.numpy().astype(np.uint8)

    # Applica le maschere
    masks = apply_masks(detected_image, segmented_image)

    with open('./prompts_no_desc.json', encoding='utf-8') as f:
        prompts = json.load(f).get('prompts')

    # Open CLIP text decoder
    device = Utils.get_device()
    model_manager = OpenClipModelManager(device)
    image_processor = ImageProcessorClip(model_manager.preprocess, device)
    text_processor = TextProcessor(model_manager.tokenizer, device)
    clip_inference = ClipInference(model_manager, image_processor, text_processor)

    # estrazione feature vectors per ogni maschera
    image_features = {}
    #plt.imshow(masks[10])
    #plt.show()

    for _, value in masks.items():
        inference = clip_inference.run_inference(value, prompts)
        inference, index = inference.squeeze(0).max(dim=0)
        img = image_processor.load_and_process_image(value)
        image_features[prompts[index]] = model_manager.encode_image(img)

    #print(image_features)

if __name__ == "__main__":
    main()
