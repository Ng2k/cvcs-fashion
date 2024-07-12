"""
@author Nicola Guerra
@author Davide Lupo
@author Francesco Mancinelli
"""
import json
import time
import torch
from torch.utils.data import DataLoader

from polyvore_dataset import PolyvoreDataset

from src.feature_extractor.classes.open_clip import OpenClip
from src.feature_extractor.feature_extractor import FeatureExtractor
from src.utils import Utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight = "./src/feature_extractor/finetuned_clip.pt"

def write_out_images(out_images: dict) -> None:
    for key in list(out_images.keys()):
        with open(f"./dataset/polyvore_feature_vectors/{key}.json", "w") as json_file:
            json.dump({ "images": out_images[key] }, json_file, indent=4)

def main():
    start_time = time.time()

    images_dir = "dataset/polyvore_40000/"
    data_loader_options = {
        "batch_size": 1024,
        "shuffle": False,
        "num_workers": 8
    }
    dataset = PolyvoreDataset(images_dir, transform = None)
    dataloader = DataLoader(dataset, **data_loader_options)

    prompt_list = Utils.get_prompt_list()
    feature_extractor = FeatureExtractor(OpenClip())
    out_images_json = {}

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}/{len(dataloader)} elaborato.")
        batch_size = len(batch["img_path"])

        for j in range(batch_size):
            print(f"Elemento {j+1}/{batch_size} del Batch {i+1}")
            img_array: torch.Tensor = batch["img_array"][j].cpu().numpy()
            img_path: str = batch["img_path"][j]

            inferences = feature_extractor.clip_model.run_inference(img_array, prompt_list)
            _, index_label = inferences.squeeze(0).max(dim=0)

            index_label_value = index_label.item()
            if index_label_value not in out_images_json:
                out_images_json[index_label_value] = []

            out_images_json[index_label.item()].append({
                "path": img_path,
                "features": feature_extractor.decode(img_array).tolist(),
                "label_clip_index": index_label_value,
                "label_clip": prompt_list[index_label_value],
            })

        print("---------------------")

    write_out_images(out_images_json)

    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time} secondi")

if __name__ == "__main__":
    main()