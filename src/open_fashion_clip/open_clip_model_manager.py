"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
from os import path

import torch
import open_clip

class OpenClipModelManager:
    """
    Classe per la gestione del modello di open-clip
    """
    _MODEL_NAME = "ViT-B/32"
    _TOKENAZER_NAME = "ViT-B-32"
    _WEIGHTS_PATH = path.join(path.dirname(__file__), "finetuned_clip.pt")

    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.preprocess, self.tokenizer = self.load_model(
            self._MODEL_NAME, self._WEIGHTS_PATH, self._TOKENAZER_NAME
        )

    def load_model(
        self,
        model_name: str,
        weights_url: str,
        tokenizer: str
    ) -> tuple:
        """
        Ritorna il modello, la funzione di pre-processing dell'immagine
        e il tokenizer per il prompt di testo

        Parametri:
        ----------
            model_name (str): nome del modello
            weights_url (str): percorso del file di stato del modello
            tokenizer (str): nome del tokenizer

        Ritorno:
        ----------
            tuple: modello, funzione di pre-processing, tokenizer
        """
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        state_dict = torch.load(weights_url, map_location=self.device)
        model.load_state_dict(state_dict["CLIP"])
        model = model.eval().requires_grad_(False).to(self.device)
        tokenizer = open_clip.get_tokenizer(tokenizer)
        return (model, preprocess, tokenizer)

    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Funzione di encode dell'immagine

        Args:
        ----------
            img_tensor (torch.Tensor): tensor dell'immagine

        Returns:
        ----------
            torch.Tensor | any: tensor dell'immagine codificato
        """
        with torch.no_grad():
            return self.model.encode_image(img_tensor.unsqueeze(0))

    def encode_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """Funzione di encode del prompt di testo

        Args:
        ----------
            text_tensor (torch.Tensor): tensor del prompt di testo

        Returns:
        ----------
            torch.Tensor | any: tensor del prompt di testo codificato
        """
        with torch.no_grad():
            return self.model.encode_text(text_tensor)

    def calculate_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Calcola la similarità tra le features dell'immagine e del testo

        Args:
            image_features (torch.Tensor): tensor delle features dell'immagine
            text_features (torch.Tensor): tensor delle features del testo

        Returns:
            torch.Tensor | any: tensor della similarità
        """
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (100.0 * image_features @ text_features.T).softmax(dim=-1)
