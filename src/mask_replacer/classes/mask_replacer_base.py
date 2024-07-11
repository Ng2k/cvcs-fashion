"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

import json
from typing import List, Tuple
import numpy as np
import torch

from src.mask_replacer.interfaces.interface_mask_replacer import IMaskReplacer
from src.utils import Utils

class MaskReplacerBase(IMaskReplacer):
    """Classe base per modulo di sostituzione delle maschere
    """

    def _find_mask_to_replace(self) -> Tuple[np.ndarray, int]:
        mask_list = self._mask_list
        feature_extractor = self._feature_extractor
        features = [feature_extractor.decode(mask) for mask in mask_list]

        similarity_matrix = self._similarity_controller.compute_similarity(features)
        mean_similarity = similarity_matrix.mean(dim=0)
        _, index_lowest_similarity = mean_similarity.squeeze(0).min(dim=0)

        return (mask_list[index_lowest_similarity], index_lowest_similarity.item())

    def _find_label_mask_to_replace(self, prompt_list: List[str]) ->  int:
        mask_list = self._mask_list
        label_mapper = self._label_mapper
        map_mask_to_label = label_mapper.map_labels_to_masks(mask_list, prompt_list)
        _, idx_mask = self._find_mask_to_replace()
        return map_mask_to_label[idx_mask]["idx_label"]

    def replace_mask(self, prompt_list: List[str]) -> List[dict]:
        features = self._feature_mask_list

        _, idx_mask = self._find_mask_to_replace()
        features_to_keep = [feature for i, feature in enumerate(features) if i != idx_mask]

        mean_similarity_list = []
        label_mask = self._find_label_mask_to_replace(prompt_list)
        polyvore_image_features = Utils.get_polyvore_image_feature(label_mask)
        for img in polyvore_image_features:
            img_features = torch.tensor(img["features"]).to(Utils.get_device())
            similarity_matrix = calculate_similarity(features_to_keep + [img_features])
            mean_similarity = similarity_matrix.mean(dim=0)
            mean_similarity_list.append(mean_similarity[-1])
