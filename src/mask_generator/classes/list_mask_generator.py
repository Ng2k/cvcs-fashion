"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from typing import List
import numpy as np

from mask_generator.classes.base_mask_generator import BaseMaskGenerator
from mask_generator.types.list_mask_type import ListMaskType

class ListMaskGenerator(BaseMaskGenerator):
    """Classe per la generazione delle maschere con output una lista di dictionaries
    """
    def __init__(self):
        super().__init__()
        self.output_data: List[ListMaskType] = []

    def _generate_output(
        self,
        input_image: np.ndarray,
        mask: np.ndarray,
        label: int,
    ) -> List[ListMaskType]:
        self.output_data.append({
            "general_label": label,
            "mask": self._apply_mask_to_image(input_image, mask)
        })
        return self.output_data

    def generate_masks(
        self,
        input_image: np.ndarray,
        segmented_image: np.ndarray
    ) -> List[ListMaskType] :
        return super().generate_masks(input_image, segmented_image)
