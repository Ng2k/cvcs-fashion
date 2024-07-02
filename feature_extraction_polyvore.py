"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
import json
import time

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import open_clip

from src.features_extractor.feature_extractor import FeatureExtractor
from src.features_extractor.open_clip.model import OpenClipModel
from polyvore_dataset import PolyvoreDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight = "./src/features_extractor/open_clip/finetuned_clip.pt"

def load_model():
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    state_dict = torch.load(weight, map_location=device)
    clip_model.load_state_dict(state_dict['CLIP'])
    clip_model = clip_model.eval().requires_grad_(False).to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return clip_model, preprocess, tokenizer

def get_prompts(path: str) -> list[str]:
    with open(path, "r") as prompts_file:
        prompts: list = json.load(prompts_file).get('prompts')

    return prompts

def get_out_images(path: str) -> list:
    try:
        with open(path, "r") as out_file:
            out_images: list = json.load(out_file).get('images')
    except Exception as _:
        out_images = []

    return out_images

def write_out_images(path: str, out_images: list) -> None:
    with open(path, "w") as json_file:
        json.dump({ "images": out_images }, json_file, indent=4)

def main():
    start_time = time.time()

    images_dir = "dataset/polyvore_64"
    data_loader_options = {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 4
    }
    dataset = PolyvoreDataset(images_dir, transform = None)
    dataloader = DataLoader(dataset, **data_loader_options)

    prompts_path = "./prompts_no_desc.json"
    out_images_path = "./feature_vectors_polyvore.json"
    prompts = get_prompts(prompts_path)
    out_images = get_out_images(out_images_path)
    feature_extractor = FeatureExtractor(OpenClipModel())

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}/{len(dataloader)} elaborato.")
        batch_size = len(batch["img_path"])

        for j in range(batch_size):
            print(f"Elemento {j+1}/{batch_size} del Batch {i+1}")
            img_array: torch.Tensor = batch["img_array"][j].cpu().numpy()
            img_path: str = batch["img_path"][j]

            img_tensor = feature_extractor.model.load_and_process_image(img_array)
            inferences = feature_extractor.model.run_inference(
                img_tensor,
                prompts
            )
            _, index_label = inferences.squeeze(0).max(dim=0)
            out_images.append({
                "path": img_path,
                "features": img_tensor.tolist(),
                "label_clip_index": index_label.item(),
                "label_clip": prompts[index_label],
                "label_general_index": -1,
                "label_general": ""
            })

        print("---------------------")

    write_out_images(path=out_images_path, out_images=out_images)

    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time} secondi")

if __name__ == "__main__":
    main()