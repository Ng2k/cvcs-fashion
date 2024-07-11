"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""

from abc import ABC, abstractmethod
import numpy as np
import torch

from src.feature_extractor.interfaces.interface_inference_model import IInferenceModel

class IClipModel(IInferenceModel, ABC):
    """Interfaccia per modello CLIP
    """

    _MODEL_NAME: str
    _TOKENAZER_NAME: str
    _WEIGHTS_PATH: str

    @abstractmethod
    def _load_and_process_image(self, image: np.ndarray) -> torch.Tensor:
        """Carica e processa un'immagine

        Args:
        -------
            image (np.ndarray): vettore immagine

        Returns:
        -------
            torch.Tensor: immagine processata
        """

    @abstractmethod
    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Funzione di encode dell'immagine

        Args:
        ----------
            image (np.ndarray): rappresentazione numpy dell'immagine

        Returns:
        ----------
            torch.Tensor: tensor dell'immagine codificato
        """

    @abstractmethod
    def _encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """Funzione di encode del prompt di testo

        Args:
        ----------
            text_tensor (torch.Tensor): tensor del prompt di testo

        Returns:
        ----------
            torch.Tensor | any: tensor del prompt di testo codificato
        """

    @abstractmethod
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Estrae le features da un'immagine

        Args:
        -------
            image (np.ndarray): immagine da cui estrarre le features

        Returns:
        -------
            torch.Tensor: tensore di features
        """
