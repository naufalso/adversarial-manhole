import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from adv_manhole.texture_mapping.depth_utils import process_surface_coordinates

from PIL import Image
from datasets import load_dataset

def load_hf_dataset(dataset_name:str, batch_size:int=8, cache_dir:str='') -> dict:
    data_iterable = {}
    
    dataset = load_dataset("naufalso/carla_hd", cache_dir="")
    
    filtered_columns_dataset = dataset.select_columns(
        ["rgb", "raw_depth", "camera_config"]
    )
    
    for data_type, dataset in filtered_columns_dataset.items():
        data_iterable[data_type] = dataset.to_iterable_dataset()
        data_iterable[data_type] = data_iterable[data_type].map(
            lambda example: {
                "local_surface_coors": transforms.ToTensor()(
                    process_surface_coordinates(example["raw_depth"], example["camera_config"])
                ),
                "rgb": transforms.ToTensor()(example["rgb"]),
            },
            remove_columns=["raw_depth", "camera_config"],
        )
        data_iterable[data_type] = DataLoader(
            data_iterable[data_type],
            batch_size=batch_size,
        )
        
    return data_iterable, filtered_columns_dataset

def image_loader(image_name, loader):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to("cuda", torch.float)

def load_manhole_set(manhole_set_path:str, image_size:int):
    candidate_images = []
    
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # scale imported image
        transforms.ToTensor()]
    )  # transform it into a torch tensor
    
    for filename in os.listdir(manhole_set_path):
        candidate_img = image_loader(f"{manhole_set_path}/{filename}", loader)
        candidate_images.append(candidate_img)
        
    candidate_images = torch.cat(candidate_images, dim=0)
    return candidate_images
    