"""
Implementazione dell'interfaccia SSDModel per il modello SSD di NVIDIA.

Fornisce metodi per caricare il modello SSD di NVIDIA e le relative utility di elaborazione.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

import torch
from src.ssd.ssd_model import SSDModel

class NVidiaSSDModel(SSDModel):
    """
    Classe concreta per il modelli SSD di nvidia.
    """

    _CONFIDENCE: float = 0.40

    def load_model(self):
        """Carica dall'hub di torch il modello SSD di nvidia

        Returns:
        --------
            any: modello SSD di nvidia
        """
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model

    def load_utils(self):
        """Carica dall'hub di torch le utils del modello SSD di nvidia

        Returns:
        --------
            any: utils di nvidia
        """
        return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
