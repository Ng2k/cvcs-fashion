"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
import torch

class ISimilarityFunction(ABC):
    """Interfaccia per il calcolo della similarità tra vettori di features
    """
    @abstractmethod
    def compute_similarity(self, features) -> torch.Tensor:
        """Ritorna una matrice di similarità tra i vettori di features

        Args:
        -------
            features1 (list[np.ndarray]): vettore di tutte le features delle immagini
        
        Returns:
        -------
            torch.Tensor: tensore con le similarità tra vettori di features
        """
