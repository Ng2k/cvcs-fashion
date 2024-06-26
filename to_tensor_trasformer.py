"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
import torch
import open_clip

from src.utils import Utils

class ToTensor():
    """Converti le immagini PIL in tensori e normalizzale."""
    def __init__(self):
        pass

    def to_tensor(self, img) -> torch.Tensor:
        """
        Args:
            img (PIL.Image): Immagine da trasformare.
        
        Returns:
            Tensor: Immagine trasformata.
        """
        device = Utils.get_device()
        weight = "./src/features_extractor/open_clip/finetuned_clip.pt"

        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
        state_dict = torch.load(weight, map_location=device)
        clip_model.load_state_dict(state_dict['CLIP'])
        clip_model = clip_model.eval().requires_grad_(False).to(device)

        return preprocess(img).to(device)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Immagine da trasformare.
        
        Returns:
            Tensor: Immagine trasformata.
        """
        return self.to_tensor(img)
