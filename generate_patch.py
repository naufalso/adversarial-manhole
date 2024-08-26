import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb

import torch.optim as optim
from torchvision import transforms

from utils.config_loader import load_yaml
from utils.data_loader import load_hf_dataset, load_manhole_set

from adv_manhole.models import load_models, ModelType
from adv_manhole.attack.losses import AdvManholeLosses
from adv_manhole.attack.naturalness import AdvContentLoss
from adv_manhole.texture_mapping.depth_utils import process_surface_coordinates
from adv_manhole.texture_mapping.depth_mapping import DepthTextureMapping
from adv_manhole.attack.framework import AdvManholeFramework

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help='Specify Config Path', default='configs/generate_patch.yml')

def main():
    args = parser.parse_args()
    
    # Load Configs file
    cfg = load_yaml(args.config_path)
    
    # Set cuda device
    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])
    
    # Load dataset
    batch_size=cfg['dataset']['batch_size']
    dataset, filtered_dataset = load_hf_dataset(
        dataset_name=cfg['dataset']['name'],
        batch_size=batch_size,
        cache_dir=cfg['dataset']['cache_dir']
    )

    # Load manhole candidate
    manhole_set = load_manhole_set(
        manhole_set_path=cfg['manhole_set']['manhole_candidate_path'],
        image_size=cfg['manhole_set']['image_size']
    )
    
    # Load MonoDepth2 model
    mde_model = load_models(ModelType.MDE, cfg['model']['mde_model'])

    # Load DDRNet model
    ss_model = load_models(ModelType.SS, cfg['model']['ss_model'])
    
    # Load Loss
    adv_content_loss = AdvContentLoss(
        candidate_images=manhole_set,
    )
    
    adversarial_losses = AdvManholeLosses(
        adv_content_loss=adv_content_loss,
        mde_loss_weight=cfg['patches']['loss_init_weight']['mde_loss_weight'],
        ss_ua_loss_weight=cfg['patches']['loss_init_weight']['ss_ua_loss_weight'],
        ss_ta_loss_weight=cfg['patches']['loss_init_weight']['ss_ta_loss_weight'],
        tv_loss_weight=cfg['patches']['loss_init_weight']['tv_loss_weight'],
        content_loss_weight=cfg['patches']['loss_init_weight']['content_loss_weight'],
        background_loss_weight=cfg['patches']['loss_init_weight']['background_loss_weight']
    )
    
    # Define depth planar mapping
    depth_planar_mapping = DepthTextureMapping(
        random_scale=(0.0, 0.01),
        with_circle_mask=True,
        device=device
    )    
    
    # Define patch texture
    texture_res = cfg['patches']['texture']['texture_resolution']
    adversarial_texture = torch.rand((3, texture_res, texture_res)).cuda()
    patch_texture_var = torch.nn.Parameter(adversarial_texture, requires_grad=True)
    
    # Get batch total
    filtered_columns_dataset = filtered_dataset.select_columns(
        ["rgb", "raw_depth", "camera_config"]
    )

    train_total_batch = len(filtered_columns_dataset["train"]) // batch_size + 1 if len(filtered_columns_dataset["train"]) % batch_size != 0 else 0
    val_total_batch = len(filtered_columns_dataset["validation"]) // batch_size + 1 if len(filtered_columns_dataset["validation"]) % batch_size != 0 else 0
    test_total_batch = len(filtered_columns_dataset["test"]) // batch_size + 1 if len(filtered_columns_dataset["test"]) % batch_size != 0 else 0
    
    # Define augmentation
    brightness = cfg['patches']['texture']['texture_augmentation']['brightness']
    contrast = cfg['patches']['texture']['texture_augmentation']['contrast']
    texture_augmentation = transforms.Compose(
        [
            transforms.ColorJitter(brightness=cfg['patches']['texture']['texture_augmentation']['brightness'], contrast=contrast),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    output_augmentation = transforms.Compose(
        [
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
        ]
    )
    
    # Define Optimizer
    optimizer = optim.Adam([patch_texture_var], lr=cfg['parameter']['learning_rate'])
    
    adv_manhole_instance = AdvManholeFramework(
        optimizer=optimizer,
        mde_model=mde_model,
        ss_model=ss_model,
        loss=adversarial_losses,
        patch_texture_var=patch_texture_var,
        depth_planar_mapping=depth_planar_mapping,
        texture_augmentation=texture_augmentation,
        output_augmentation=output_augmentation,
        device=device
    )
    
    adv_manhole_instance.train(
        epochs=cfg['parameter']['epochs'],
        dataset=dataset,
        train_total_batch=train_total_batch,
        val_total_batch=val_total_batch,
        log_prediction_every=cfg['log']['log_prediction'],
        log_name=cfg['log']['log_name']
    )
    
if __name__ == "__main__":
    main()