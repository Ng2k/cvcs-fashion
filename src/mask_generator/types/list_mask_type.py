"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass
import torch

from mask_generator.interfaces.interface_mask_type import IMaskType

@dataclass
class ListMaskType(IMaskType):
    """Classe per definizione tipo di dato per le maschere

    ```python
    {
        "general_label": int,
        "mask": np.ndarray,
    }
    ```

    Attributes:
    ------
        general_label (int): label della segmentazione
        mask (torch.Tensor): tensore della maschera
    """
    general_label: int
    mask: torch.Tensor
