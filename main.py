"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from PIL import Image
from matplotlib import pyplot as plt

from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

from src.image_processor import ImageProcessor

from src.segmentation.segformer_b2_clothes import SegformerB2Clothes
from src.segmentation.clothes_segmantion import ClothesSegmentation

def image_resize(image_url: str, max_size: int) -> Image:
    return ImageProcessor.resize_image(Image.open(image_url), max_size)

def ssd_detection(image: Image) -> Image:
    """Ritorna una tupla con le cordinate del bounding box con la personaù

        Args:
        -------
            image_url : str
                Percorso immagine

        Return
        -------
            tuple
                tupla con le coordinate dei bbox
    """
    #plt.imshow(image)
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    return ssd_model.detect_person_in_image(image)

    #plt.imshow(ssd_model.detect_person_in_image(image_url))

def segmentation(image : Image) -> Image:
    segmentation_model = ClothesSegmentation(SegformerB2Clothes())
    return segmentation_model.apply_segmentation(image)

def main():
    #ssd_detection("./test_2.jpg")
    resized_image = image_resize("./img_test.jpg", 300)
    segmented_image = segmentation(resized_image)
    plt.imshow(segmented_image)

    #image = ssd_detection(resized_image)
    #segmentation(image)

if __name__ == "__main__":
    ssd_detection("./img_test.jpg")
