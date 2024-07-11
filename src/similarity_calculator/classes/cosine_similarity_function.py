"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from typing import List
import torch

from src.similarity_calculator.interfaces.interface_similarity_function import ISimilarityFunction

class CosineSimilarityFunction(ISimilarityFunction):
    """Implementazione del calcolo della similarità coseno tra vettori di features
    """
    def compute_similarity(self, features: List[torch.Tensor]) -> torch.Tensor:
        features_tensor = torch.cat(features, dim = 0)
        return torch.cosine_similarity(
            features_tensor.unsqueeze(1),
            features_tensor.unsqueeze(0),
            dim = -1
        )
