import torch
import argparse

from utils.config_loader import load_yaml
from utils.data_loader import load_hf_dataset, load_manhole_set

from adv_manhole.models import load_models, ModelType
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
        cache_dir=cfg['dataset']['cache_dir'],
        filter_set='eval',
        selected_columns=["rgb", "raw_depth", "camera_config", "semantic"]
    )
    
    # Load manhole candidate
    manhole_set = load_manhole_set(
        manhole_set_path=cfg['manhole_set']['manhole_candidate_path'],
        image_size=cfg['manhole_set']['image_size'],
        adversarial_sample_images=cfg['manhole_set']['adversarial_sample_images'],
        is_eval=True
    )
    
    # Load MonoDepth2 model
    mde_model = load_models(ModelType.MDE, cfg['model']['mde_model'])

    # Load DDRNet model
    ss_model = load_models(ModelType.SS, cfg['model']['ss_model'])
    
    # Define depth planar mapping
    depth_planar_mapping = DepthTextureMapping(
        random_scale=(0.0, 0.01),
        with_circle_mask=True,
        device=device
    )
    
    train_total_batch = len(filtered_dataset["train"]) // batch_size + 1 if len(filtered_dataset["train"]) % batch_size != 0 else 0
    val_total_batch = len(filtered_dataset["validation"]) // batch_size + 1 if len(filtered_dataset["validation"]) % batch_size != 0 else 0
    test_total_batch = len(filtered_dataset["test"]) // batch_size + 1 if len(filtered_dataset["test"]) % batch_size != 0 else 0
    
    adv_manhole_instance = AdvManholeFramework(
        mde_model=mde_model,
        ss_model=ss_model,
        depth_planar_mapping=depth_planar_mapping,
        device=device
    )
    
    adv_manhole_instance.evaluate(
        dataset=dataset,
        total_batch=[train_total_batch, val_total_batch, test_total_batch],
        manhole_set=manhole_set,
        log_name=cfg['log']['log_name']
    )
    
if __name__ == "__main__":
    main()