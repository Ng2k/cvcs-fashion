"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch

class IInferenceModel(ABC):
    """Interfaccia per modello di inferenza per le immagini
    """

    @abstractmethod
    def _load_model(self) -> tuple:
        """
        Ritorna il modello, la funzione di pre-processing dell'immagine
        e il tokenizer per il prompt di testo

        Returns:
        -------
            tuple: modello, funzione di pre-processing, tokenizer
        """

    @abstractmethod
    def run_inference(self, image: np.ndarray, prompt_list: List[str]) -> torch.Tensor:
        """Esegue l'inferenza

        Args:
        --------
            image_tensor (torch.Tensor): percorso dell'immagine
            prompt_list (List[str]): lista di prompt di testo

        Returns:
        --------
            torch.Tensor: tensore delle inference dell'immagine
        """
