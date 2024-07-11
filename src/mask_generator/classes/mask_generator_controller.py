"""
Controller per algoritmo di creazione delle maschere

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import numpy as np

from mask_generator.interfaces.interface_mask_generator import IMaskGenerator
from mask_generator.types.mask_type import IMaskType

class MaskGeneratorController:
    """Controller per la classe di generazione delle maschere
    """
    def __init__(self, mask_generator: IMaskGenerator) -> None:
        self.mask_generator = mask_generator

    def generate_masks(
        self,
        input_image: np.ndarray,
        segmented_image: np.ndarray
    ) -> IMaskType:
        """Creazione maschere per l'immagine

        Args:
            input_image (np.ndarray): immagine di input
            segmented_image (np.ndarray): immagine segmentata

        Returns:
            List[IMaskType]: lista delle maschere generate
        """
        return self.mask_generator.generate_masks(input_image, segmented_image)
