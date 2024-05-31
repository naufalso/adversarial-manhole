import torch
import numpy as np

class DepthTextureMapping:
    def __init__(self, texture_res=256, tex_scale=0.5, tex_offset=[0.0, 0.0], random_scale=(0.0, 0.25), random_shift=(-0.4, 0.4), device=None):
        self.texture_res = texture_res
        self.tex_scale = tex_scale
        self.tex_offset = tex_offset
        self.random_scale = random_scale
        self.random_shift = random_shift
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def _get_texture_scale(self, batch_size: int):
        tex_scales = (
            torch.rand(batch_size) * (self.random_scale[1] - self.random_scale[0])
            + self.random_scale[0]
        ) + self.tex_scale

        return tex_scales.to(self.device)
    
    def _get_texture_offset(self, batch_size: int):
        x_offset = (
            torch.rand(batch_size) * (self.random_shift[1] - self.random_shift[0])
            + self.random_shift[0]
            + self.tex_offset[0]
        ) 

        y_offset = (
            torch.rand(batch_size) * (self.random_shift[1] - self.random_shift[0])
            + self.random_shift[0]
            + self.tex_offset[1]
            - (self.tex_scale * 0.5)  # Shift the offset to center the texture
        )  

        xyz_offsets = torch.stack([x_offset, y_offset, torch.zeros(batch_size)], dim=1)

        return xyz_offsets.to(self.device)

    def _create_circle_mask(self, height, width, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(width/2), int(height/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], width-center[0], height-center[1])

        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return torch.tensor(mask).unsqueeze(0).float().to(self.device)

    
    def mapped_texture_mask(self, surface_xyz, tex_scales, xyz_offsets):
        batch_image_size = [surface_xyz.shape[0], surface_xyz.shape[2], surface_xyz.shape[3]]

        x_min = xyz_offsets[:, 0].view(-1, 1, 1).broadcast_to(*batch_image_size)
        x_max = (xyz_offsets[:, 0] + tex_scales).view(-1, 1, 1).broadcast_to(*batch_image_size)
        y_min = xyz_offsets[:, 1].view(-1, 1, 1).broadcast_to(*batch_image_size)
        y_max = (xyz_offsets[:, 1] + tex_scales).view(-1, 1, 1).broadcast_to(*batch_image_size)

        # Mask the centerized local coordinates given the texture size
        texture_mask = torch.where(
            (surface_xyz[:, 0] > x_min)
            & (surface_xyz[:, 0] < x_max)
            & (surface_xyz[:, 1] > y_min)
            & (surface_xyz[:, 1] < y_max),
            1.0,
            0.0,
        )

        return texture_mask.unsqueeze(1)
    
    def depth_texture_mapping(self, texture, surface_xyz, tex_scales, xyz_offsets):
        """
        Maps a depth texture onto a surface based on UV coordinates.

        Args:
            texture (Tensor): The texture to be mapped onto the surface.
            surface_xyz (Tensor): The surface coordinates of the object.

        Returns:
            Tensor: The rendered image with the texture mapped onto the surface.
        """
        # Convert to channel last
        surface_xyz = surface_xyz.permute(0, 2, 3, 1)

        texture = texture.permute(0, 2, 3, 1)
        channel_count = texture.shape[-1]
        texture_flat = texture.reshape(-1, channel_count)

        print(texture_flat.shape)

        max_uv_idx = self.texture_res**2 - 1

        xyz_offsets_broadcasted = xyz_offsets.view(-1, 1, 1, 3).broadcast_to(*surface_xyz.shape)
        tex_scales_broadcasted = tex_scales.view(-1, 1, 1, 1).broadcast_to(*surface_xyz.shape)

        print(surface_xyz.shape, xyz_offsets.shape, tex_scales_broadcasted.shape)

        # Get the UV coordinates by modulating the surface coordinates with the texture size
        surface_xyz_mod = (surface_xyz - xyz_offsets_broadcasted) % tex_scales_broadcasted # tex_scales.view(-1, 1, 1, 1)
        surface_xyz_mod = surface_xyz_mod / tex_scales_broadcasted # tex_scales.view(-1, 1, 1, 1)

        # Get the UV indices by multiplying the UV coordinates with the texture resolution
        uv_idx = surface_xyz_mod * self.texture_res
        uv_idx = torch.round(uv_idx).to(torch.int64)
        uv_idx = torch.clamp(uv_idx, 0, self.texture_res - 1)

        # Get the UV indices for each axis
        uv_idx_z = uv_idx[..., [0, 1]]

        # Invert the z uv indices
        uv_idx_z[..., 0] = self.texture_res - uv_idx_z[..., 0]

        # Get the UV indices for each axis flattened
        uv_idx_z_flat = (uv_idx_z[..., 0] * self.texture_res + uv_idx_z[..., 1]).clamp_max(
            max_uv_idx
        )

        # Get the rendered images for each axis
        rendered_img_z = texture_flat[uv_idx_z_flat.flatten(), :].reshape(
            (surface_xyz_mod.shape[0], surface_xyz_mod.shape[1],  surface_xyz_mod.shape[2], channel_count)
        )

        # Reverse the channel last to channel first
        rendered_img_z = rendered_img_z.permute(0, 3, 1, 2)

        return rendered_img_z