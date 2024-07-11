"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from typing import List
from functools import partial
import numpy as np

from src.feature_extractor.interfaces.interface_inference_model import IInferenceModel
from src.mask_generator.types.mask_type import MaskType

class LabelMapper():
    """Classe per il mapping delle label custom alle maschere in formato List
    """
    def __init__(self, inference_model: IInferenceModel):
        self.inference_model = inference_model

    def _mapper(self, mask: np.ndarray, prompt_list: List[str]):
        """Funzione mapper

        Args:
            mask (_type_): _description_
            prompt_list (_type_): _description_
        """
        inferences = self.inference_model.run_inference(mask, prompt_list)
        _, index_label = inferences.squeeze(0).max(dim=0)
        return {
            "idx_label": index_label.item(),
            "label": prompt_list[index_label.item()],
            "mask": mask
        }

    def map_labels_to_masks(
        self,
        masks_list: List[MaskType],
        prompt_list: List[str]
    ) -> List[dict]:
        return list(map(
            partial(self._mapper, prompt_list=prompt_list),
            list(map(lambda x: x['mask'], masks_list))
        ))
