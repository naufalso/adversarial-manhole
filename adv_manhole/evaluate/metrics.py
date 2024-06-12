import torch

def error_distance_region(predicted_depth, original_depth, texture_masks, road_masks, distance_threshold=0.25, output_region=False):
    """
    Calculate the relative error distance mean and ratio for depth maps in road area.

    Args:
        predicted_depth (torch.Tensor, shape (N, 1, H, W)): Predicted depth map.
        original_depth (torch.Tensor, shape (N, 1, H, W)): Original depth map.
        texture_masks (torch.Tensor, shape (N, 1, H, W)): Binary mask indicating texture regions.
        road_masks (torch.Tensor, shape (N, 1, H, W)): Binary mask indicating road regions.
        distance_threshold (float, optional): Threshold for error distance. Defaults to 0.1.
        output_region (bool, optional): Whether to output the error distance region. Defaults to False.

    Returns:
        Dict of floats, the relative error distance mean and ratio.
    """


    # Calculate abs rel error distance
    # predicted_depth = 1 / (predicted_disp + 1e-6)
    # original_depth = 1 / (original_disp + 1e-6)
    error = torch.abs(original_depth - predicted_depth) / original_depth

    # Error distance masked
    error_masked = error[texture_masks > 0.5]

    # Calculate error distance region
    error_distance_region = (error > distance_threshold).float() * road_masks

    # Calculate the error area for each sample
    error_area = error_distance_region.squeeze(1).sum(dim=(1, 2))

    # Make sure the error area shape is same as batch size
    # assert error_area.shape == road_masks.shape[0], f"Error area shape {error_area.shape} != {road_masks.shape[0]}"

    # Calculate the texture area for each sample
    texture_area = texture_masks.squeeze(1).sum(dim=(1, 2))

    # Make sure the texture area shape is same as batch size
    # assert texture_area.shape == road_masks.shape[0], f"Texture area shape {texture_area.shape} != {road_masks.shape[0]}"

    # Calculate the ratio of error distance region to texture area
    ratio = error_area / texture_area

    if output_region:
        return {
            "ed_mean": error_masked.mean(),
            "ed_ratio": ratio.mean(),
            "ed_region": error_distance_region
        }

    return {
        "ed_mean": error_masked.mean(),
        "ed_ratio": ratio.mean()
    }


def asr_segmentation_region(predicted_semantic, original_semantic, original_index, target_indices, texture_masks, road_masks, output_region=False):
    """
    Calculate the Attack Success Rate (ASR) for semantic segmentation for both untargeted and targeted attacks in road area.

    Args:
        predicted_semantic: torch.Tensor, shape (N, C, H, W), the predicted semantic segmentation.
        original_semantic: torch.Tensor, shape (N, C, H, W), the original semantic segmentation.
        original_index: int, the original class index.
        target_indices: list of int, the target class indices.
        texture_masks: torch.Tensor, shape (N, 1, H, W), the texture masks.
        road_masks: torch.Tensor, shape (N, 1, H, W), the road masks.
        output_region: bool, optional, whether to output the region. Defaults to False.

    Returns:
        Dict of floats, the untargeted ASR and targeted ASR in road area.
    """

    # Get the predicted class indices
    predicted_indices = predicted_semantic.argmax(dim=1)
    
    # Get original class indicies
    original_indicies = original_semantic.argmax(dim=1)

    # Get the texture masks
    texture_masks = texture_masks.squeeze(1)

    # Get the road masks
    road_masks = road_masks.squeeze(1)

    # Get original road masks
    original_road_masks = (original_indicies == original_index).float()

    # Calculate the untargeted ASR in texture area
    untargeted_asr = (predicted_indices != original_index).float()
    untargeted_asr_mean = untargeted_asr[texture_masks > 0.5].mean()
    
    # Calculate the targeted ASR in texture area
    targeted_asr = torch.zeros(predicted_indices.shape, device=predicted_indices.device)
    for target_index in target_indices:
        targeted_asr += (predicted_indices == target_index).float()
    targeted_asr_mean = targeted_asr[texture_masks > 0.5].mean()

    # Calculate the texture area
    texture_area = texture_masks.squeeze(1).sum(dim=(1, 2))

    # Calculate the area from untargeted ASR in road area
    untargeted_asr_road = (untargeted_asr * road_masks * original_road_masks)
    untargeted_area = untargeted_asr_road.squeeze(1).sum(dim=(1, 2))

    # Calculate the ratio of area from targeted ASR in road area
    targeted_area_road = (targeted_asr * road_masks * original_road_masks)
    targeted_area = targeted_area_road.squeeze(1).sum(dim=(1, 2))

    # Calculate the ratio of area from untargeted ASR in road area to texture area
    untargeted_ratio = (untargeted_area / texture_area)

    # Calculate the ratio of area from targeted ASR in road area to texture area
    targeted_ratio = targeted_area / texture_area

    if output_region:
        return {
            "asr_ua_mean": untargeted_asr_mean,
            "asr_ta_mean": targeted_asr_mean,
            "ua_ratio": untargeted_ratio.mean(),
            "ta_ratio": targeted_ratio.mean(),
            "ua_region": untargeted_asr_road,
            "ta_region": targeted_area_road
        }

    return {
        "asr_ua_mean": untargeted_asr_mean,
        "asr_ta_mean": targeted_asr_mean,
        "ua_ratio": untargeted_ratio.mean(),
        "ta_ratio": targeted_ratio.mean()
    }