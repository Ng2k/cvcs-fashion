"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import numpy as np

from mask_generator.classes.base_mask_generator import BaseMaskGenerator
from mask_generator.types.dict_mask_type import DictMaskType

class DictMaskGenerator(BaseMaskGenerator):
    """Classe per la generazione delle maschere con output una lista di dictionaries
    """

    def __init__(self):
        super().__init__()
        self.output_data = {}

    def _generate_output(
        self,
        input_image: np.ndarray,
        mask: np.ndarray,
        label: int,
    ) -> DictMaskType:
        self.output_data[label] = (input_image * mask[:, :, None]).astype(np.uint8)
        return self.output_data

    def generate_masks(
        self,
        input_image: np.ndarray,
        segmented_image: np.ndarray
    ) -> DictMaskType:
        return super().generate_masks(input_image, segmented_image)
