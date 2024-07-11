"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
import numpy as np

from feature_extractor.interfaces.interface_clip_model import IClipModel

class FeatureExtractor():
    """Classe base per estrarre features da immagini
    """
    def __init__(self, clip_model: IClipModel):
        self.clip_model = clip_model

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Estrae le features da un'immagine

        Args:
        -------
            image (np.ndarray): immagine da cui estrarre le features
            prompt_list (List[str]): lista di prompt di testo

        Returns:
        -------
            torch.Tensor: tensor di features
        """
        return self.clip_model.extract_features(image)
