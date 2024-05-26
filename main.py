"""Main file.

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from src.ssd.nvidia_ssd_model import NVidiaSSDModel
from src.ssd.single_shot_detector import SingleShotDetector

def ssd_detection(image_url: str) -> tuple:
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
    ssd_model = SingleShotDetector(NVidiaSSDModel())
    ssd_model.detect_person_in_image(image_url)

if __name__ == "__main__":
    ssd_detection("./img_test.jpg")
