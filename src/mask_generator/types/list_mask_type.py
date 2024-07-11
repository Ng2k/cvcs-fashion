"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass
from typing import List
import torch

from src.mask_generator.types.mask_type import IMaskType

@dataclass
class MaskDict:
    """Classe per definizione tipo di dato per le maschere
    """
    general_label: int
    mask: torch.Tensor

@dataclass
class ListMaskType(IMaskType):
    """Classe per definizione tipo di dato per le maschere

    ```python
    [
        {
            "general_label": int,
            "mask": np.ndarray,
        },
        ...
    ]
    ```

    Attributes:
    ------
        general_label (int): label della segmentazione
        mask (torch.Tensor): tensore della maschera
    """
    def __init__(self, data: List[MaskDict]):
        self.data = data
