"""
Implementazione dell'interfaccia SSDModel per il modello SSD di NVIDIA.

Fornisce metodi per caricare il modello SSD di NVIDIA e le relative utility di elaborazione.

@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""

import torch
import numpy as np

from src.ssd.ssd_model import SSDModel

class NVidiaSSDModel(SSDModel):
    """
    Classe concreta per il modelli SSD di nvidia.
    """

    _CONFIDENCE: float = 0.40

    def __init__(self):
        """
        Inizializza un nuovo oggetto NVidiaSSDModel.
        """
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_ssd_processing_utils'
        )

        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def _prepare_image(self, image_url: str) -> np.ndarray:
        """Ritorna l'immagine come un array numpy.

        Args:
        -------
            image_url (str): percorso immagine

        Returns:
        -------
            np.ndarray: array numpy che rappresenta l'immagine
        """
        return self.utils.prepare_input(image_url)

    def _prepare_tensor(self, images_list: list) -> torch.Tensor:
        """Ritorna l'immagine come un tensore PyTorch.

        Args:
        -------
            images_list (list): lista di immagini

        Returns:
        -------
            torch.Tensor: tensore PyTorch che rappresenta l'immagine
        """
        return self.utils.prepare_tensor(images_list)

    def load_image(self, image_url: str) -> dict:
        image_numpy = self._prepare_image(image_url)
        return {
            "image_numpy": image_numpy,
            "image_tensor": self._prepare_tensor([image_numpy])
        }

    def find_best_bboxes(self, image_tensor: torch.Tensor) -> list:
        with torch.no_grad():
            detections_batch = self.model(image_tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [
            self.utils.pick_best(results, self._CONFIDENCE) for results in results_per_input
        ]

        return best_results_per_input
