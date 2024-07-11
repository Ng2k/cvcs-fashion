"""
@Author Nicola Guerra
@Author: Davide Lupo
@Author: Francesco Mancinelli
"""
import torch
from torch import nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from src.clothes_segmentation.interfaces.interface_segmentation_model import ISegmentationModel

class SegformerB2Clothes(ISegmentationModel):
    """
    Classe concreta per il modello di segmentazione SegFormer_B2_Clothes.
    """

    def __init__(self):
        """
        Inizializza un nuovo oggetto SegformerB2Clothes.
        """
        model_name = "mattmdjaga/segformer_b2_clothes"
        self._processor = SegformerImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    def apply_segmentation(self, image: Image) -> torch.Tensor:
        inputs = self._processor(images=image, return_tensors="pt")

        outputs = self._model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        return pred_seg
