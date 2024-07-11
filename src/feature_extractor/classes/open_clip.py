"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from os import path
from typing import List

import torch
import numpy as np
import open_clip
from PIL import Image

from feature_extractor.interfaces.interface_clip_model import IClipModel

from src.utils import Utils

class OpenClip(IClipModel):
    """Implementazione Open Clip
    """
    _MODEL_NAME = "ViT-B/32"
    _TOKENAZER_NAME = "ViT-B-32"
    _WEIGHTS_PATH = path.join(path.dirname(__file__), "finetuned_clip.pt")

    def __init__(self):
        self.device = Utils.get_device()
        self.model, self.preprocess, self.tokenizer = self._load_model()

    def _load_model(self) -> tuple:
        model, _, preprocess = open_clip.create_model_and_transforms(self._MODEL_NAME)
        state_dict = torch.load(self._WEIGHTS_PATH, map_location = self.device)
        model.load_state_dict(state_dict["CLIP"])
        model = model.eval().requires_grad_(False).to(self.device)
        tokenizer = open_clip.get_tokenizer(self._TOKENAZER_NAME)
        return (model, preprocess, tokenizer)

    def _load_and_process_image(self, image: np.ndarray) -> torch.Tensor:
        img = self.preprocess(Image.fromarray(image)).to(Utils.get_device())
        return img

    def _tokenize_prompts(self, prompt_list: List[str]):
        """Tokenizza i prompt di testo

        Args:
        -------
            prompts (list): lista di prompt di testo

        Returns:
        -------
            _type_: _description_
        """
        tokenized_prompts = self.tokenizer(prompt_list).to(self.device)
        return tokenized_prompts

    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self._load_and_process_image(image)
        with torch.no_grad():
            return self.model.encode_image(image_tensor.unsqueeze(0))

    def _encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_text(text_tensor)

    def _calculate_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (100.0 * image_features @ text_features.T).softmax(dim=-1)

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        return self._encode_image(image).unsqueeze(0)

    def run_inference(self, image: np.ndarray, prompt_list: List[str]) -> torch.Tensor:
        text_tensor = self._tokenize_prompts(prompt_list)

        image_features = self._encode_image(image)
        text_features = self._encode_text(text_tensor)

        text_probs = self._calculate_similarity(image_features, text_features)
        return text_probs
