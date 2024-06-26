"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
import json
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip

from polyvore_dataset import PolyvoreDataset
from to_tensor_trasformer import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight = "./src/features_extractor/open_clip/finetuned_clip.pt"

def load_model():
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    state_dict = torch.load(weight, map_location=device)
    clip_model.load_state_dict(state_dict['CLIP'])
    clip_model = clip_model.eval().requires_grad_(False).to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return clip_model, preprocess, tokenizer

def features_extraction(batch_images: torch.Tensor) -> torch.Tensor:
    """
    Estrae le caratteristiche per un batch di immagini utilizzando il modello CLIP.
    
    Args:
    --------
        batch_images (torch.Tensor): Un batch di immagini preprocessate.
    
    Returns:
    --------
        torch.Tensor: Un tensore contenente le caratteristiche estratte per ogni immagine.
    """
    clip_model, _, tokenizer = load_model()
    clip_model.to(device)
    batch_images = batch_images.to(device)

    with open("./prompts_no_desc.json", "r") as prompts_file:
        prompts = json.load(prompts_file).get('prompts')

    tokenized_prompt = tokenizer(prompts).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(batch_images)
        text_features = clip_model.encode_text(tokenized_prompt)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calcola la similarità e applica softmax per ottenere le probabilità
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, labels = similarities.max(dim=1)

    return labels, image_features

def main():
    start_time = time.time()

    images_dir = "dataset/polyvore"
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ToTensor()
    ])
    dataset = PolyvoreDataset(images_dir, transform = transform)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4
    )

    counter = 0
    for i, batch in enumerate(dataloader):
        _, feature_vectors = features_extraction(batch)
        print(f"Batch {i+1}/{len(dataloader)} elaborato.")

    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time} secondi")


if __name__ == "__main__":
    main()