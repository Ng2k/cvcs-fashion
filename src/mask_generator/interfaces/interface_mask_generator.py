"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
import numpy as np

from src.mask_generator.types.mask_type import IMaskType

class IMaskGenerator(ABC):
    """Interfaccia per il calcolo delle maschere
    """
    @abstractmethod
    def _generate_output(
        self,
        input_image: np.ndarray,
        mask: np.ndarray,
        label: int
    ) -> IMaskType:
        """Genera struttura dati di output

        Args:
        -------
            output_data: dati di output
            input_image (np.ndarray): immagine di input
            mask (np.ndarray): maschera
            label (int): label della segmentazione
        
        Returns:
        -------
            List[IMaskType]: lista delle maschere generate
        """

    @abstractmethod
    def generate_masks(
        self,
        input_image: np.ndarray,
        segmented_image: np.ndarray
    ) -> IMaskType:
        """Genera le maschere delle immagini

        Args:
        -------
            input_image (np.ndarray): immagine di input
            segmented_image (np.ndarray): immagine segmentata
        """
