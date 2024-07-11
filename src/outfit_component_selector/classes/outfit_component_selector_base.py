"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from typing import List, Tuple
import cv2
import numpy as np
import torch

from src.outfit_component_selector.interfaces.interface_outfit_component_selector import IOutfitComponentSelector
from src.utils import Utils

class OutfitComponentSelector(IOutfitComponentSelector):
    """Classe base per modulo di sostituzione delle maschere
    """

    def _find_mask_to_replace(self) -> Tuple[np.ndarray, int]:
        """Ritorna la maschera da rimpiazzare e l'indice della maschera

        Args:
        ------
            masks (IMaskType): maschere dell'immagine

        Returns:
        ------
            Tuple[np.ndarray, int]: maschera e indice della maschera da rimpiazzare
        """
        mask_list = self._mask_list
        features = self._feature_list

        similarity_matrix = self._similarity_controller.compute_similarity(features)
        mean_similarity = similarity_matrix.mean(dim=0)
        _, index_lowest_similarity = mean_similarity.squeeze(0).min(dim=0)

        return (mask_list[index_lowest_similarity], index_lowest_similarity.item())

    def _find_label(
        self,
        prompt_list: List[str],
        idx_mask_to_replace: int
    ) ->  int:
        """Ritorna la label custom della maschera da rimpiazzare

        Args:
        ------
            prompt_list (List[str]): lista delle label custom
            idx_mask_to_replace (int): indice della maschera da rimpiazzare

        Returns:
        ------
            int: indice della label custom
        """
        mask_list = self._mask_list
        label_mapper = self._label_mapper
        map_mask_to_label = label_mapper.map_labels_to_masks(mask_list, prompt_list)
        return map_mask_to_label[idx_mask_to_replace]["idx_label"]

    def find_best_fit(self, prompt_list: List[str]) -> np.ndarray:
        _, idx_mask = self._find_mask_to_replace()
        features_to_keep = [f for i, f in enumerate(self._feature_list) if i != idx_mask]
        label = self._find_label(prompt_list, idx_mask)
        polyvore_images = Utils.get_polyvore_image_feature(label)
        mean_similarity_list = [
            self._similarity_controller.compute_similarity(
                features_to_keep + [torch.tensor(img["features"]).to(Utils.get_device())]
            )
            .mean(dim=0)[-1]
            for img in polyvore_images
        ]

        _, idx_best_similarity = torch.tensor(mean_similarity_list).max(dim=0)
        return cv2.imread(polyvore_images[idx_best_similarity]["path"])
