"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from feature_extractor.interfaces.interface_inference_model import IInferenceModel
from mask_generator.types.mask_type import IMaskType

class ILabelMapper(ABC):
    """Interfaccia per il mapping delle label custom alle maschere dell'immagine
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

    @abstractmethod
    def map_labels_to_masks(self, masks: IMaskType, prompt_list: List[str]) -> List[dict]:
        """Metodo per il mapping delle label custom alle maschere dell'immagine

        Args:
        -------
            masks (IMaskType): maschere da mappare
            prompt_list (List[str]): lista delle label custom
        
        Returns:
        -------
            List[dict]: lista di dizionari con la label custom, l'indice della label e la maschera
        """
