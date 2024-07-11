"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
from dataclasses import dataclass
import torch

@dataclass
class MaskType():
    """Classe per definizione tipo di dato per le maschere
    """
    general_label: int
    mask: torch.Tensor
