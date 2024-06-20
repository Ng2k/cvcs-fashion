"""
Classe per la gestione delle maschere

@Author: Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import numpy as np

class MaskProcessor:
    """
    Classe per processare le maschere di segmentazione.
    """
    @staticmethod
    def compute_masks(input_image: np.ndarray, segmented_image: np.ndarray) -> dict:
        """
        Computa le maschere per isolare i capi di abbigliamento dalla segmentazione.

        Parametri
        ----------
            input_image : np.ndarray
                Immagine originale
            segmented_image : np.ndarray
                Immagine segmentata

        Ritorna
        -------
            ```python
            {
                label: np.ndarray
                ...
            }
            ```
                Le maschere per i capi di abbigliamento.
        """
        masks_dict = {}
        for label in np.unique(segmented_image):
            mask = np.where(segmented_image == label, 1, 0)
            if np.any(mask):
                masks_dict[label] = (input_image * mask[:, :, None]).astype(np.uint8)

        return masks_dict
