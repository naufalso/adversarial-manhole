import torch


def mapped_texture_mask(
    surface_xyz,
    tex_size=1.0,
    tex_shift=[0.0, 0.0, 0.0],
):
    """
    Generate a texture mask based on the given surface coordinates.

    Args:
        surface_xyz (torch.Tensor): The surface coordinates.
        tex_size (float, optional): The size of the texture. Defaults to 1.0.
        tex_shift (list[float], optional): The shift of the texture. Defaults to [0.0, 0.0, 0.0].

    Returns:
        torch.Tensor: The texture mask.
    """

    x_offset = tex_shift[0]
    y_offset = tex_shift[1]

    # Mask the centerized local coordinates given the texture size
    texture_mask = torch.where(
        (surface_xyz[0, ...] > x_offset)
        & (surface_xyz[0, ...] < x_offset + tex_size)
        & (surface_xyz[1, ...] > y_offset)
        & (surface_xyz[1, ...] < y_offset + tex_size),
        1.0,
        0.0,
    )

    return texture_mask


def depth_texture_mapping(
    texture,
    surface_xyz,
    tex_size=1.0,
    tex_shift=[0.0, 0.0, 0.0],
    texture_res=256,
):
    """
    Maps a depth texture onto a surface based on UV coordinates.

    Args:
        texture (Tensor): The texture to be mapped onto the surface.
        surface_xyz (Tensor): The surface coordinates of the object.
        tex_size (float, optional): The size of the texture. Defaults to 1.0.
        tex_shift (list[float], optional): The shift of the texture. Defaults to [0.0, 0.0, 0.0].
        texture_res (int, optional): The resolution of the texture. Defaults to 256.

    Returns:
        Tensor: The rendered image with the texture mapped onto the surface.
    """

    # Convert to channel last
    surface_xyz = surface_xyz.permute(1, 2, 0)
    # normal_mask = normal_mask.permute(1, 2, 0)
    texture_flat = texture.permute(1, 2, 0).view(-1, 3)
    max_uv_idx = texture_res**2 - 1

    # Get the UV coordinates by modulating the surface coordinates with the texture size
    surface_xyz_mod = (surface_xyz - tex_shift) % tex_size
    surface_xyz_mod = surface_xyz_mod / tex_size

    # Get the UV indices by multiplying the UV coordinates with the texture resolution
    uv_idx = surface_xyz_mod * texture_res
    uv_idx = torch.round(uv_idx).to(torch.int64)
    uv_idx = torch.clamp(uv_idx, 0, texture_res - 1)

    # Get the UV indices for each axis
    uv_idx_z = uv_idx[..., [0, 1]]

    # Invert the z uv indices
    uv_idx_z[..., 0] = texture_res - uv_idx_z[..., 0]

    # Get the UV indices for each axis flattened
    uv_idx_z_flat = (uv_idx_z[..., 0] * texture_res + uv_idx_z[..., 1]).clamp_max(
        max_uv_idx
    )

    # Get the rendered images for each axis
    rendered_img_z = texture_flat[uv_idx_z_flat.flatten(), :].reshape(
        surface_xyz_mod.shape
    )

    # Reverse the channel last to channel first
    rendered_img_z = rendered_img_z.permute(2, 0, 1)

    return rendered_img_z
