"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass
import numpy as np

from mask_generator.interfaces.interface_mask_type import IMaskType

@dataclass
class DictMaskType(IMaskType):
    """Classe per definizione tipo di dato per le maschere

    ```python
    {
        "<general_label>": <mask>,
    }
    ```

    Attributes:
    ------
        general_label (int): label della segmentazione
        mask (np.ndarray): tensore della maschera
    """
    general_label: np.ndarray
