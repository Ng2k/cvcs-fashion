"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import json
from typing import List
import torch

class Utils():
    """Classe con funzioni di utiliti generali
    """
    @staticmethod
    def get_device() -> str:
        """
        Restituisce il device su cui eseguire il modello.

        Returns:
        -------
            str: Il device su cui eseguire il modello.
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def get_prompt_list() -> List[str]:
        """Restituisce la lista delle label custom

        Returns:
        -------
            List[str]: lista delle label custom
        """
        with open(
            "./prompts_no_desc.json",
            mode = "r",
            encoding="utf-8"
        ) as prompts_file:
            return json.load(prompts_file).get('prompts')

    @staticmethod
    def get_polyvore_image_feature(label: int) -> List[dict]:
        """Restituisce le feature delle immagini di Polyvore

        Args:
        -------
            label (int): label dell'immagine

        Returns:
        -------
            List[dict]: lista delle immagini
        """
        try:
            with open(
                f"./dataset/polyvore_feature_vectors/{label}.json",
                mode = "r",
                encoding="utf-8"
            ) as img_files:
                images = json.load(img_files).get('images')
        except FileNotFoundError:
            images = []

        return images
