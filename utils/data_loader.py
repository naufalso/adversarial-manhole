import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from adv_manhole.texture_mapping.depth_utils import process_surface_coordinates, process_depth

from PIL import Image
from datasets import load_dataset

from typing import Optional

def load_hf_dataset(
    dataset_name:str, 
    batch_size:int=8, 
    cache_dir:str='',
    filter_set:str='train',
    selected_columns:list=[]
) -> dict:
    data_iterable = {}
    
    dataset = load_dataset(dataset_name, cache_dir="")
    
#     filtered_columns_dataset = dataset.select_columns(
#         ["rgb", "raw_depth", "camera_config"]
#     )

    filtered_columns_dataset = dataset.select_columns(
        selected_columns
    )
    
    data_iterable = _filter_set(
        batch_size=batch_size,
        filtered_columns_dataset=filtered_columns_dataset,
        filter_set=filter_set
    )
        
    return data_iterable, filtered_columns_dataset

def _filter_set(batch_size, filtered_columns_dataset, filter_set:str='train'):
    data_iterable = {}
    
    if filter_set == 'train':
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
    if filter_set == 'eval':
        for data_type, dataset in filtered_columns_dataset.items():
            data_iterable[data_type] = dataset.to_iterable_dataset()
            data_iterable[data_type] = data_iterable[data_type].map(
                lambda example: {
                    "local_surface_coors": transforms.ToTensor()(
                        process_surface_coordinates(example["raw_depth"], example["camera_config"])
                    ),
                    "rgb": transforms.ToTensor()(example["rgb"]),
                    "road_mask": transforms.ToTensor()(
                        np.any(np.array(example['semantic']) == np.array((128, 64, 128)), axis=-1).astype(np.float32)
                    ),
                    "depth": transforms.ToTensor()(
                        process_depth(example["raw_depth"]) * 1000.0 # Convert to meters
                    )
                },
                remove_columns=["raw_depth", "camera_config", "semantic"],
            )
            data_iterable[data_type] = DataLoader(
                data_iterable[data_type],
                batch_size=batch_size,
            )
    
    return data_iterable

def image_loader(image_name, loader):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to("cuda", torch.float)

def load_manhole_set(
    manhole_set_path:str, image_size:int, adversarial_sample_images=None, is_eval:bool=False
):
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # scale imported image
        transforms.ToTensor()]
    )  # transform it into a torch tensor

    if is_eval:
        manhole_images = {}
        
        for key, path in adversarial_sample_images.items():
            manhole_images[key] = image_loader(path, loader)

        manhole_images['random'] = torch.rand_like(manhole_images['normal'])
        
        return manhole_images
    else:
        candidate_images = []
        
        for idx, filename in enumerate(os.listdir(manhole_set_path)):
            #print(filename)
            candidate_img = image_loader(f"{manhole_set_path}/{filename}", loader)
            candidate_images.append(candidate_img)
                
        candidate_images = torch.cat(candidate_images, dim=0)
    
        return candidate_images