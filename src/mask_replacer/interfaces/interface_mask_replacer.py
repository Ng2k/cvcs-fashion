"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from src.label_mapper.classes.label_mapper import LabelMapper
from src.mask_generator.types.mask_type import MaskType
from src.feature_extractor.feature_extractor import FeatureExtractor
from src.similarity_calculator.similarity_controller import SimilarityController

class IMaskReplacer(ABC):
    """Interfaccia per modulo di sostituzione delle maschere"""

    _FEATURE_IMAGES_POLYVORE_PATH = "./dataset/polyvore_feature_vectors/"

    def __init__(
        self,
        mask_list: List[MaskType],
        feature_extractor: FeatureExtractor,
        similarity_controller: SimilarityController,
        label_mapper: LabelMapper
    ) -> None:
        self._mask_list = mask_list

        self._feature_extractor: FeatureExtractor = feature_extractor
        self._feature_mask_list = [
            self._feature_extractor.decode(m["mask"]) for m in self._mask_list
        ]

        self._similarity_controller = similarity_controller
        self._label_mapper = label_mapper

    @abstractmethod
    def _find_mask_to_replace(self) -> Tuple[np.ndarray, int]:
        """Ritorna la maschera da rimpiazzare e l'indice della maschera

        Args:
        ------
            masks (IMaskType): maschere dell'immagine

        Returns:
        ------
            Tuple[np.ndarray, int]: maschera e indice della maschera da rimpiazzare
        """
    def _find_label_mask_to_replace(self, prompt_list: List[str]) -> int:
        """Ritorna la label custom della maschera da rimpiazzare

        Args:
        ------
            prompt_list (List[str]): lista delle label custom

        Returns:
        ------
            int: indice della label custom
        """

    def replace_mask(self, prompt_list: List[str]) -> np.ndarray:
        """Sostituisce la maschera dell'immagine con un indumento preso dal dataset

        Args:
        ------
            prompt_list (List[str]): lista delle features delle maschere

        Returns:
        ------
            np.ndarray: maschera dell'indumento da sostituire
        """
