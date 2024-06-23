"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

import numpy as np
import torch

from src.similarity_calculator.interface_similarity_function import ISimilarityFunction

class CosineSimilarityFunction(ISimilarityFunction):
    """Implementazione del calcolo della similarità coseno tra vettori di features
    """
    def compute_similarity(self, features: list[np.ndarray]) -> torch.Tensor:
        features_tensor = torch.cat(features, dim = 0)
        return torch.cosine_similarity(
            features_tensor.unsqueeze(1),
            features_tensor.unsqueeze(0),
            dim = -1
        )
