"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
from typing import List
import torch

class ISimilarityFunction(ABC):
    """Interfaccia per il calcolo della similarità tra vettori di features
    """
    @abstractmethod
    def compute_similarity(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Ritorna una matrice di similarità tra i vettori di features

        Args:
        -------
            features1 (List[torch.Tensor]): vettore di tutte le features delle immagini
        
        Returns:
        -------
            torch.Tensor: tensore con le similarità tra vettori di features
        """
