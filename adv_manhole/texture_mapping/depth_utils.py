import math
import numpy as np
import torch


def process_depth(depth_img_raw: np.ndarray) -> np.ndarray:
    """
    Process the depth image to obtain normalized depth values.

    Args:
        depth_img_raw (np.ndarray): The raw depth image.

    Returns:
        np.ndarray: The normalized depth image.
    """
    # Load the depth image
    depth_img_raw = np.array(depth_img_raw) / 255.0

    # Process carla depth image into correct depth values
    depth_img = (
        (depth_img_raw[:, :, 0])
        + (depth_img_raw[:, :, 1] * 256.0)
        + (depth_img_raw[:, :, 2] * 256.0 * 256.0)
    )
    normalized_depth_img = depth_img / (256.0 * 256.0 * 256.0 - 1.0)

    return normalized_depth_img


# This function is used to convert the depth image to local coordinates
def depth_to_local_coordinates(
    normalized_depth_img: np.ndarray, camera_config: dict
) -> np.ndarray:
    """
    Convert depth image to local 3D coordinates.

    Args:
        normalized_depth_img (np.ndarray): Normalized depth image.
        camera_config (dict): Camera configuration parameters.

    Returns:
        np.ndarray: Local 3D coordinates in Unreal coordinate system.
    """

    # Camera Intrinsic
    # (Intrinsic) K Matrix
    Cin = np.identity(3)
    Cin[0, 2] = camera_config["image_width"] / 2.0
    Cin[1, 2] = camera_config["image_height"] / 2.0
    Cin[0, 0] = Cin[1, 1] = camera_config["image_width"] / (
        2.0 * math.tan(camera_config["fov"] * math.pi / 360.0)
    )

    # # (Extrinsic) Transform Matrix [No need to use it]
    # Cex = np.array(camera_config["transform_matrix"])

    # 2D Pixel Coordinates [u, v, 1]
    pixel_length = camera_config["image_width"] * camera_config["image_height"]
    u_coord, v_coord = np.meshgrid(
        np.arange(camera_config["image_width"]),
        np.arange(camera_config["image_height"]),
    )
    u_coord_flat = np.reshape(u_coord, pixel_length)
    v_coord_flat = np.reshape(v_coord, pixel_length)
    normalized_depth_flat = np.reshape(normalized_depth_img, pixel_length)

    uv_matrix = np.stack([u_coord_flat, v_coord_flat, np.ones_like(u_coord_flat)])

    # 3D Local Coordinates = [x, y, z] (camera)
    local_3d_coordinate = np.dot(np.linalg.inv(Cin), uv_matrix)
    local_3d_coordinate *= normalized_depth_flat * 1000.0  # Convert to meters

    # New we must change from "standard" camera coordinate system (the same used by OpenCV) to Unreal coordinate system

    #    z             ^ z
    #   /              |
    #  +-------> x to  |
    #  |               | . x
    #  |               |/
    #  v y             +-------> y

    # Convert the camera coordinate to unreal coordinate: [x, y, z] -> [x, z, y]
    local_3d_coordinate_ue4 = np.array(
        [local_3d_coordinate[2], local_3d_coordinate[0], local_3d_coordinate[1] * -1.0]
    ).T

    # Reshape to image size
    local_3d_coordinate_ue4 = np.reshape(
        local_3d_coordinate_ue4,
        (camera_config["image_height"], camera_config["image_width"], 3),
    )

    # Centerize the local coordinates
    x_coord_min = np.min(local_3d_coordinate_ue4[:, :, 0])
    z_coord_min = np.min(local_3d_coordinate_ue4[:, :, 2])

    centerized_local_3d_coordinate_ue4 = local_3d_coordinate_ue4 - np.array(
        [x_coord_min, 0.0, z_coord_min]
    )

    # Convert from meters to centimeters
    centerized_local_3d_coordinate_ue4 *= 100.0

    # Convert to float32
    centerized_local_3d_coordinate_ue4 = centerized_local_3d_coordinate_ue4.astype(
        np.float32
    )

    return centerized_local_3d_coordinate_ue4


def process_surface_coordinates(raw_depth, camera_config):
    # Process the depth image
    processed_depth_img = process_depth(raw_depth)

    # Convert the depth image to local coordinates
    surface_xyz = depth_to_local_coordinates(processed_depth_img, camera_config)

    return surface_xyz

def disp_to_depth(disparity, min_depth=0.1, max_depth=100.0):
    """
    Converts the disparity map to depth map.

    Args:
        disparity (torch.Tensor): The disparity map.
        min_depth (float): The minimum depth value.
        max_depth (float): The maximum depth value.

    Returns:
        torch.Tensor: The depth map.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    scaled_disp = min_disp + (max_disp - min_disp) * disparity
    depth = 1 / scaled_disp

    return depth
    

def median_scaling(prediction_depth, gt_depth, reference_masks = None):
    """
    Scales the predicted depth map to match the ground truth median depth.

    Args:
        prediction_depth (torch.Tensor): The predicted depth map.
        gt_depth (torch.Tensor): The ground truth depth map.
        reference_masks (torch.Tensor): The reference masks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scaled depth map and the scaling ratio.
    """

    if reference_masks is None:
        reference_masks = torch.ones_like(prediction_depth)

    gt_depth[gt_depth == 0] = 1e-3
    ratio = torch.median(gt_depth[reference_masks > 0.5]) / torch.median(prediction_depth[reference_masks > 0.5])
    scaled_depth = prediction_depth * ratio

    return scaled_depth, ratio
