"""
@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import numpy as np

from mask_generator.interfaces.interface_mask_generator import IMaskGenerator
from mask_generator.types.mask_type import IMaskType
from mask_generator.utility import MaskUtility

class BaseMaskGenerator(IMaskGenerator):
    """Classe per la generazione delle maschere
    """
    def __init__(self):
        self.output_data = None

    def _generate_mask_for_label(
        self,
        segmented_image: np.ndarray,
        label: int,
        last_mask=None
    ) -> np.ndarray:
        """Genrazione maschera per la label

        Args:
        --------
            segmented_image (np.ndarray): immagine segmentata
            label (int): label della segmentazione
            last_mask (np.ndarray, optional): ultima maschera. None valore di default.

        Returns:
        --------
            np.ndarray: maschera per la label
        """
        mask = np.where(segmented_image == label, 1, 0)
        if last_mask is not None:
            mask += last_mask
        return mask

    def _is_valid_mask(
        self,
        mask: np.ndarray,
        label: int,
        left_shoe_index: int
    ) -> bool:
        """Verifica se la maschera è valida

        Args:
        --------
            mask (_type_): _description_
            label (_type_): _description_
            left_shoe_index (_type_): _description_

        Returns:
        --------
            bool: True se la maschera è valida, False altrimenti
        """
        return np.any(mask) and label != left_shoe_index

    def _apply_mask_to_image(
        self,
        input_image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Applica la maschera all'immagine

        Args:
            input_image (np.ndarray): _description_
            mask (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return (input_image * mask[:, :, None]).astype(np.uint8)

    def generate_masks(
        self,
        input_image: np.ndarray,
        segmented_image: np.ndarray,
    ) -> IMaskType:
        right_shoe_idx, left_shoe_idx = MaskUtility.get_shoes_index()
        last_mask = None

        for label in np.unique(segmented_image):
            if label in MaskUtility.get_label_black_list():
                continue

            last_mask = None if label != right_shoe_idx else last_mask
            mask = self._generate_mask_for_label(segmented_image, label, last_mask)

            if not self._is_valid_mask(mask, label, left_shoe_idx):
                continue

            self.output_data = self._generate_output(input_image, mask, label)
            last_mask = mask

        return self.output_data
