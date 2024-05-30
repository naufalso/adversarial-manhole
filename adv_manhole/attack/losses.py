import torch
from typing import List, Tuple, Optional
from adv_manhole.attack.naturalness import AdvContentLoss


class AdvManholeLosses:

    def __init__(
        self,
        adv_content_loss: Optional[AdvContentLoss] = None,
        mde_loss_weight: float = 1.0,
        ss_ua_loss_weight: float = 1.0,
        ss_ta_loss_weight: float = 1.0,
        tv_loss_weight: float = 0.25,
        content_loss_weight: float = 0.001,
        background_loss_weight: float = 0.0,
        loss_function: str = "bce",
    ):
        """
        Initializes the Losses object.

        Args:
            mde_loss_weight (float): Weight for the MDE loss. Default is 1.0.
            ss_ua_loss_weight (float): Weight for the SS-UA loss. Default is 1.0.
            ss_ta_loss_weight (float): Weight for the SS-TA loss. Default is 1.0.
            tv_loss_weight (float): Weight for the TV loss. Default is 0.25.
            content_loss_weight (float): Weight for the content loss. Default is 0.25.
            background_loss_weight (float): Weight for the background loss. Default is 0.0.
            loss_function (str): Loss function to use. Supported values are "bce" (binary cross entropy) and "mse" (mean squared error). Default is "bce".

        Raises:
            ValueError: If an unsupported loss function is provided.

        """
        self.adv_content_loss = adv_content_loss

        self.mde_loss_weight = mde_loss_weight
        self.ss_ua_loss_weight = ss_ua_loss_weight
        self.ss_ta_loss_weight = ss_ta_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.content_loss_weight = content_loss_weight
        self.background_loss_weight = background_loss_weight

        # Set the loss function
        if loss_function == "bce":
            self.loss_function = torch.nn.functional.binary_cross_entropy
        elif loss_function == "mse":
            self.loss_function = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        self.mse_loss = torch.nn.functional.mse_loss
        self.mae_loss = torch.nn.functional.l1_loss

    def tv_loss(self, texture: torch.Tensor) -> torch.Tensor:
        """
        Compute the smoothness loss for the texture.

        Args:
            texture (torch.Tensor): The texture to compute the smoothness loss for.

        Returns:
            torch.Tensor: The smoothness loss for the texture.
        """

        # Compute the gradient of the texture patch
        texture_dx = texture - torch.roll(texture, shifts=1, dims=1)
        texture_dy = texture - torch.roll(texture, shifts=1, dims=2)

        # Compute MAE loss for the texture
        loss = self.mae_loss(texture_dx, torch.zeros_like(texture_dx)) + self.mae_loss(
            texture_dy, torch.zeros_like(texture_dy)
        )

        return loss * self.tv_loss_weight


    def background_loss(
        self,
        patched_images: torch.Tensor,
        backgrounds: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the background loss for the patched images.

        Args:
            patched_images (torch.Tensor): The patched images.
            backgrounds (torch.Tensor): The background images.
            masks (torch.Tensor): The masks for the patched images.

        Returns:
            torch.Tensor: The background loss for the patched images.
        """

        # Compute the background loss and ignore the masked regions
        loss = self.mse_loss(patched_images, backgrounds, reduction="none")
        valid_indices = masks > 0.5
        valid_loss = loss[valid_indices]
        final_loss = torch.mean(valid_loss)

        return final_loss * self.background_loss_weight

    def adversarial_mde_loss(
        self,
        predicted_disparities: torch.Tensor,
        target_disparities: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the adversarial loss for the monocular depth estimation model.

        Args:
            predicted_disparities (torch.Tensor): The predicted disparities.
            target_disparities (torch.Tensor): The target disparities.
            masks (torch.Tensor): The masks for the disparities.

        Returns:
            torch.Tensor: The adversarial loss for the monocular depth estimation model.
        """

        # Compute the BCE loss and ignore the masked regions
        loss = self.loss_function(
            predicted_disparities, target_disparities, reduction="none"
        )

        valid_indices = masks > 0.5
        valid_loss = loss[valid_indices]
        final_loss = torch.mean(valid_loss)

        return final_loss * self.mde_loss_weight

    def adversarial_ss_loss(
        self,
        predicted_masks: torch.Tensor,
        original_index: int,
        target_indices: List[int],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the adversarial loss for the semantic segmentation model.

        Args:
            predicted_masks (torch.Tensor): The predicted masks.
            original_index (int): The original index of the target labels.
            target_indices (torch.Tensor): The target labels.
            masks (torch.Tensor): The masks for the masks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The untargeted and targeted adversarial losses.
        """

        original_probs = predicted_masks[:, original_index]
        target_probs = predicted_masks[:, target_indices]

        valid_indices = masks.squeeze(1) > 0.5

        # Untargeted loss
        original_loss = self.loss_function(
            original_probs, torch.zeros_like(original_probs), reduction="none"
        )
        valid_original_loss = original_loss[valid_indices]
        final_original_loss = torch.mean(valid_original_loss)

        # Take the highest probability of the target labels
        largest_target_probs = torch.max(target_probs, dim=1)[0]

        # Targeted loss
        target_loss = self.loss_function(
            largest_target_probs,
            torch.ones_like(largest_target_probs),
            reduction="none",
        )

        valid_target_loss = target_loss[valid_indices]
        final_target_loss = torch.mean(valid_target_loss)

        return (
            final_original_loss * self.ss_ua_loss_weight,
            final_target_loss * self.ss_ta_loss_weight,
        )
    
    def adversarial_content_loss(self, texture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the adversarial content loss.

        Args:
            input_images (torch.Tensor): The input images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The adversarial content loss and the index of the candidate image.
        """
        if self.adv_content_loss is None:
            raise ValueError("AdvContentLoss is not provided during initialization")
        
        content_loss, min_idx = self.adv_content_loss(texture.unsqueeze(0))

        return content_loss * self.content_loss_weight, min_idx


    def __call__(self, texture, patched_images, masks, backgrounds, predicted_disparities, target_disparities, predicted_masks, original_index, target_indices):
        """
        Compute the total loss for the adversarial manhole attack.

        Args:
            texture (torch.Tensor): The texture.
            patched_images (torch.Tensor): The patched images.
            masks (torch.Tensor): The masks for the patched images.
            backgrounds (torch.Tensor): The background images.
            predicted_disparities (torch.Tensor): The predicted disparities.
            target_disparities (torch.Tensor): The target disparities.
            predicted_masks (torch.Tensor): The predicted masks.
            original_index (int): The original index of the target labels.
            target_indices (torch.Tensor): The target labels.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss and the individual losses.
        """

        # MDE loss
        if self.mde_loss_weight > 0:
            mde_loss = self.adversarial_mde_loss(
                predicted_disparities, target_disparities, masks
            )
        else:
            mde_loss = 0.0

        # SS-UA and SS-TA losses
        if self.ss_ua_loss_weight > 0 or self.ss_ta_loss_weight > 0:
            ss_ua_loss, ss_ta_loss = self.adversarial_ss_loss(
                predicted_masks, original_index, target_indices, masks
            )
        else:
            ss_ua_loss = 0.0
            ss_ta_loss = 0.0


        # TV loss
        if self.tv_loss_weight > 0:
            tv_loss = self.tv_loss(texture)
        else:
            tv_loss = 0.0

        # Content loss
        if self.content_loss_weight > 0:
            content_loss, min_idx = self.adversarial_content_loss(texture)
        else:
            content_loss = 0.0
            min_idx = 0

        # Background loss
        if self.background_loss_weight > 0:
            bg_loss = self.background_loss(patched_images, backgrounds, masks.repeat(1, 3, 1, 1))
        else:
            bg_loss = 0.0

        # Compute the total loss
        total_loss = mde_loss + ss_ua_loss + ss_ta_loss + tv_loss + content_loss + bg_loss

        return {
            "total_loss": total_loss,
            "mde_loss": mde_loss,
            "ss_ua_loss": ss_ua_loss,
            "ss_ta_loss": ss_ta_loss,
            "tv_loss": tv_loss,
            "content_loss": content_loss,
            "min_idx": min_idx,  # "min_idx" is the index of the candidate image with the minimum content loss
            "background_loss": bg_loss,
        }

