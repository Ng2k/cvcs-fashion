"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
import numpy as np

from src.open_fashion_clip.open_clip_model_manager import OpenClipModelManager
from src.open_fashion_clip.image_processor_clip import ImageProcessorClip
from src.open_fashion_clip.text_processor import TextProcessor

class ClipInference:
    """
    Classe per l'inferenza di CLIP
    """
    def __init__(
        self,
        model_manager: OpenClipModelManager,
        image_processor: ImageProcessorClip,
        text_processor: TextProcessor
    ):
        self.model_manager = model_manager
        self.image_processor = image_processor
        self.text_processor = text_processor

    def run_inference(self, image: np.ndarray, prompts: list[str]) -> torch.Tensor:
        """Esegue l'inferenza di CLIP

        Parametri:
        --------
            image (np.ndarray): percorso dell'immagine
            prompts (list[str]): lista di prompt di testo

        Returns:
        --------
            _type_: _description_
        """
        img_tensor = self.image_processor.load_and_process_image(image)
        text_tensor = self.text_processor.tokenize_prompts(prompts)

        image_features = self.model_manager.encode_image(img_tensor)
        text_features = self.model_manager.encode_text(text_tensor)

        text_probs = self.model_manager.calculate_similarity(
            image_features, text_features
        )
        return text_probs
