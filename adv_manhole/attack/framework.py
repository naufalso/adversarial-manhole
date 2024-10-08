import os
import torch
import json
import random
import pandas as pd
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from adv_manhole.models.model_mde import ModelMDE
from adv_manhole.models.model_ss import ModelSS
from adv_manhole.attack.losses import AdvManholeLosses

from adv_manhole.texture_mapping.depth_mapping import DepthTextureMapping
from adv_manhole.attack.losses import AdvManholeLosses

from adv_manhole.evaluate.metrics import error_distance_region, asr_segmentation_region
from adv_manhole.texture_mapping.depth_utils import disp_to_depth, median_scaling

class AdvManholeFramework:

    def __init__(
        self,
        mde_model: ModelMDE,
        ss_model: ModelSS,
        depth_planar_mapping:DepthTextureMapping,
        optimizer=None,
        loss: AdvManholeLosses=None,
        patch_texture_var=None,
        texture_augmentation=None,
        output_augmentation=None,
        device=None
    ):
        self.optimizer = optimizer
        self.mde_model = mde_model
        self.ss_model = ss_model
        self.loss = loss
        self.patch_texture_var = patch_texture_var
        self.depth_planar_mapping = depth_planar_mapping
        self.texture_augmentation = texture_augmentation
        self.output_augmentation = output_augmentation
        self.device = device
        
        self.original_index = 0 # Road
        self.target_indices = [3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        
    def forward(
        self,
        batch, 
        patch_texture_var, 
        depth_planar_mapping: DepthTextureMapping, 
        adversarial_losses: AdvManholeLosses, 
    ):
        rgb = batch["rgb"].to(self.device)
        local_surface_coors = batch["local_surface_coors"].to(self.device)

        current_batch_size = rgb.shape[0]

        # Repeat the texture to match the batch size
        batched_texture = patch_texture_var.unsqueeze(0).repeat(
            current_batch_size, 1, 1, 1
        )

        # Augment the texture with color jitter
        augmented_texture = torch.stack(
            [
                self.texture_augmentation(batched_texture[i])
                for i in range(current_batch_size)
            ]
        )

        final_images, texture_masks = depth_planar_mapping(
            augmented_texture, local_surface_coors, rgb, current_batch_size
        )

        # Apply the output augmentation
        augmented_final_images = torch.stack(
            [
                self.output_augmentation(final_images[i])
                for i in range(current_batch_size)
            ]
        )

        # Predict the depth and semantic segmentation
        predicted_disp = self.mde_model(augmented_final_images)
        predicted_semantic = self.ss_model(augmented_final_images)

        losses = adversarial_losses(
            patch_texture_var,
            final_images,
            texture_masks,
            rgb,
            predicted_disp,
            torch.ones_like(predicted_disp),
            predicted_semantic,
            self.original_index,
            self.target_indices
        )

        return {
            "loss": losses,
            "final_images": final_images,
            "augmented_final_images": augmented_final_images,
            "texture_masks": texture_masks,
            "predicted_disp": predicted_disp,
            "predicted_semantic": predicted_semantic,
        }
    
    def forward_eval(
        self,
        batch,
        patch_texture = None,
        tex_scales = None,
        xyz_offsets = None
    ):
        rgb = batch["rgb"].to(self.device)
        local_surface_coors = batch["local_surface_coors"].to(self.device)

        current_batch_size = rgb.shape[0]

        if patch_texture is not None:
            # Repeat the texture to match the batch size
            batched_texture = patch_texture.repeat(
                current_batch_size, 1, 1, 1
            )

            final_images, texture_masks = self.depth_planar_mapping(
                batched_texture, local_surface_coors, rgb, current_batch_size, tex_scales=tex_scales, xyz_offsets=xyz_offsets, random_scale=False, random_shift=False
            )

            # Predict the depth and semantic segmentation
            predicted_disp = self.mde_model(final_images)
            predicted_semantic = self.ss_model(final_images)

            return predicted_disp, predicted_semantic, texture_masks
        else:
            predicted_disp = self.mde_model(rgb)
            predicted_semantic = self.ss_model(rgb)
        
            return predicted_disp, predicted_semantic, None
    
    def evaluate(
        self,
        dataset,
        total_batch:list,
        manhole_set,
        log_name:str
    ):
        eval_metrics = {
            'train': defaultdict(list),
            'validation': defaultdict(list),
            'test': defaultdict(list)
        }

        for datatype, total in zip(['train', 'validation', 'test'], total_batch):
            with tqdm(
                dataset[datatype], total=total, desc=f"{datatype} Evaluation"
            ) as pbar:
                for batch in pbar:
                    metrics = {}

                    road_masks = batch["road_mask"].cuda()
                    gt_depth = batch["depth"].to('cuda')

                    with torch.no_grad():
                        current_batch_size = batch["rgb"].shape[0]
                        tex_scales = self.depth_planar_mapping._get_texture_scale(current_batch_size)
                        xyz_offsets = self.depth_planar_mapping._get_texture_offset(current_batch_size)

                        ori_predicted_disp, ori_predicted_semantic, _ = self.forward_eval(batch, tex_scales=tex_scales, xyz_offsets=xyz_offsets)
                        ori_predicted_depth = disp_to_depth(ori_predicted_disp)
                        ori_scaled_predicted_depth, ratio = median_scaling(ori_predicted_depth, gt_depth)

                        for manhole_name, manhole_image in manhole_set.items():
                            # print(key)
                            predicted_disp, predicted_semantic, texture_masks = self.forward_eval(batch, manhole_image, tex_scales=tex_scales, xyz_offsets=xyz_offsets)

                            scaled_predicted_depth = disp_to_depth(predicted_disp) * ratio

                            # Calculate the error distance
                            mde_metrics = error_distance_region(scaled_predicted_depth, ori_scaled_predicted_depth, texture_masks, road_masks, distance_threshold=0.25)

                            for mde_key in mde_metrics.keys():
                                eval_metrics[datatype][f"{manhole_name}/{mde_key}"].append(mde_metrics[mde_key].item())

                            # Calculate the ASR
                            ss_metrics = asr_segmentation_region(predicted_semantic, ori_predicted_semantic, self.original_index, self.target_indices, texture_masks, road_masks)
                            for ss_key in ss_metrics.keys():
                                eval_metrics[datatype][f"{manhole_name}/{ss_key}"].append(ss_metrics[ss_key].item())

                        metrics.update({
                            key: np.mean(eval_metrics[datatype][key])
                            for key in eval_metrics[datatype].keys()
                        })

                    pbar.set_postfix(metrics)
                    
        # Save evaluation result
        # Check if direectory is exist or not, if ot make dirs
        dir_path = os.path.join('log', log_name)
        if os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
        
        for key in eval_metrics.keys():
            json_str = pd.DataFrame(eval_metrics[key]).mean().to_json()
            final_json = json.loads(json_str)
            
            with open(os.path.join(dir_path, "eval_{}.json".format(key)), 'w') as f:
                json.dump(final_json, f, indent=4)
        
    def train(
        self,
        epochs:int,
        dataset,
        train_total_batch,
        val_total_batch,
        log_prediction_every:int,
        log_name:str
    ):
        # Check if direectory is exist or not, if ot make dirs
        dir_path = os.path.join('log', log_name)
        if os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
        
        mean_value_train_list, mean_value_eval_list = [], []
        
        train_epoch_history = defaultdict(list)
        val_epoch_history = defaultdict(list)
        for epoch in range(epochs):
            mean_value_train, mean_value_eval = 0, 0
            step_train, step_val = 0, 0
            # Train iteration
            with tqdm(
                dataset['train'], total=train_total_batch, desc=f"Train Epoch {epoch + 1}/{epochs}"
            ) as train_pbar:
                train_history = defaultdict(list)
                for batch in train_pbar:
                    self.optimizer.zero_grad()

                    results = self.forward(batch, self.patch_texture_var, self.depth_planar_mapping, self.loss)

                    # Backpropagate the loss
                    results['loss']["total_loss"].backward()
                    self.optimizer.step()

                    # Clip the texture to [0, 1]
                    self.patch_texture_var.data.clamp_(0, 1)

                    # Update the progress bar
                    train_pbar.set_postfix(
                        {
                            key: value.item() if isinstance(value, torch.Tensor) else value
                            for key, value in results['loss'].items()
                            if 'loss' in key
                        }
                    )

                    # Update the train history
                    for key, value in results['loss'].items():
                        if 'loss' in key:
                            train_history[key].append(value.item() if isinstance(value, torch.Tensor) else value)

                for key, value in train_history.items():
                    mean_value = np.mean(value)
                    train_epoch_history[key].append(mean_value)
                    # # Log the loss into disk
                    #with open(os.path.join(dir_path, "train_mean_val_step_{}.json".format(step_train)), 'w') as f:
                        #json.dump({"mean_val": mean_value}, f, indent=4)
                    
                    mean_value_train += mean_value
                    step_train += 1
                    #run.log({f"train/{key}": mean_value}, commit=False)

                # Append train log
                mean_value_train_list.append(mean_value_train/step_train)
                
                # Save the texture as image
                Image.fromarray((self.patch_texture_var.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(dir_path, f"texture_epoch_{epoch}.png")
                )

            # Validation iteration
            with tqdm(
                dataset['validation'], total=val_total_batch, desc=f"Validation {epoch + 1}/{epochs}"
            ) as val_pbar:
                val_history = defaultdict(list)
                for batch in val_pbar:
                    with torch.no_grad():
                        results = self.forward(batch, self.patch_texture_var, self.depth_planar_mapping, self.loss)

                    # Update the progress bar
                    val_pbar.set_postfix(
                        {
                            key: value.item() if isinstance(value, torch.Tensor) else value
                            for key, value in results['loss'].items()
                            if 'loss' in key
                        }
                    )

                    # Update the validation history
                    for key, value in results['loss'].items():
                        if 'loss' in key:
                            val_history[key].append(value.item() if isinstance(value, torch.Tensor) else value)

                for key, value in val_history.items():
                    mean_value = np.mean(value)
                    val_epoch_history[key].append(mean_value)
                    # # Log the loss into disk
                    #run.log({f"val/{key}": mean_value}, commit=False)
                    #with open(os.path.join(dir_path, "validation_mean_val_step_{}.json".format(step_val)), 'w') as f:
                        #json.dump({"mean_val": mean_value}, f, indent=4)

                    mean_value_eval += mean_value
                    step_val += 1
                    
                mean_value_eval_list.append(mean_value_eval/step_val)
                        
            # Log the visualizations
            if epoch % log_prediction_every == 0:
                augmented_final_images = results["augmented_final_images"]
                predicted_disp = results["predicted_disp"]
                predicted_semantic = results["predicted_semantic"]

                random_idx = random.randint(0, augmented_final_images.shape[0] - 1)

                ### Log this into disk
                disp_pred_fig = self.mde_model.plot(
                    augmented_final_images[random_idx].permute(1, 2, 0).detach().cpu().numpy(),
                    predicted_disp[random_idx],
                    save=True,
                    save_path=os.path.join(dir_path, "predicted_disp_epochs_{}.jpg".format(epochs))
                )

                semantic_pred_fig = self.ss_model.plot(
                    augmented_final_images[random_idx].permute(1, 2, 0).detach().cpu().numpy(),
                    predicted_semantic[random_idx],
                    save=True,
                    save_path=os.path.join(dir_path, "semantic_pred_epochs_{}.jpg".format(epochs))
                )
                
#                 print(type(disp_pred_fig))
#                 Image.fromarray(disp_pred_fig).save(
#                     os.path.join("log", f"disp_pred_epoch_{epoch}.png")
#                 )
                
                
#                 print(type(semantic_pred_fig))
#                 Image.fromarray(semantic_pred_fig).save(
#                     os.path.join("log", f"semantic_pred_epoch_{epoch}.png")
#                 )
                
                #disp_pred_fig = cv2.cvtColor(disp_pred_fig, cv2.COLOR_BGR2RGB)
                #sematic_pred_fig = cv2.cvtColor(sematic_pred_fig, cv2.COLOR_BGR2RGB)
                #cv2.imwrite("log/disparity_pred_epoch_{}.jpg".format(epochs), disp_pred_fig)
                #cv2.imwrite("log/sematic_pred_epoch_{}.jpg".format(epochs), semantic_pred_fig)
                
#                 run.log(
#                     {
#                         "val/disp_pred": disp_pred_fig,
#                         "val/semantic_pred": semantic_pred_fig,
#                     },
#                     commit=False
#                 )

            Image.fromarray((self.patch_texture_var.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(dir_path, f"texture_epoch_{epoch}.png")
                )
#             cv2.imwrite(os.path.join(dir_path, "texture_epochs_{}.png".format(epochs)), self.patch_texture_var.permute(1, 2, 0).detach().cpu().numpy() * 255)
    
        # Save mea value log
        log_json = {
            "train":mean_value_train_list,
            "test":mean_value_eval_list,
        }
        with open(os.path.join(dir_path, "model_loss.json"), 'w') as f:
            json.dump(log_json, f, indent=4)
        
#             ### Log Texture into disk
#             # Log the texture to wandb
#             run.log({"texture": wandb.Image(patch_texture_var.permute(1, 2, 0).detach().cpu().numpy())})