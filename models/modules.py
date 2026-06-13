from einops import reduce, rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from kornia.utils import create_meshgrid

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def get_depth_values(current_depth, n_depths, depth_interval):
    """
    get the depth values of each pixel : [depth_min, depth_max) step is depth_interval
    current_depth: (B, 1, H, W), current depth map
    n_depth: int, number of channels of depth
    depth_interval: (B, 1) or float, interval between each depth channel
    return: (B, D, H, W)
    """
    if not isinstance(depth_interval, float):
        depth_interval = rearrange(depth_interval, 'b 1 -> b 1 1 1')
    depth_min = torch.clamp_min(current_depth - n_depths/2 * depth_interval, 1e-7)
    depth_values = depth_min + depth_interval * \
                   rearrange(torch.arange(0, n_depths,
                                          device=current_depth.device,
                                          dtype=current_depth.dtype), 'd -> 1 d 1 1')
    return depth_values


def homo_warp(src_feat, proj_mat, depth_values):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device

    R = proj_mat[:, :, :3] # (B, 3, 3)
    T = proj_mat[:, :, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
    ref_grid = rearrange(ref_grid, '1 h w c -> 1 c (h w)') # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = repeat(ref_grid, 'b c x -> b c (d x)', d=D) # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T/rearrange(depth_values, 'b d h w -> b 1 (d h w)')
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory
    
    # project negative depth pixels to somewhere outside the image
    negative_depth_mask = src_grid_d[:, 2:] <= 1e-7
    src_grid_d[:, 0:1][negative_depth_mask] = W
    src_grid_d[:, 1:2][negative_depth_mask] = H
    src_grid_d[:, 2:3][negative_depth_mask] = 1

    src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:] # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0]/((W-1)/2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((H-1)/2) - 1 # scale to -1~1
    src_grid = rearrange(src_grid, 'b c (d h w) -> b d (h w) c', d=D, h=H, w=W)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = rearrange(warped_src_feat, 'b c d (h w) -> b c d h w', h=H, w=W)

    return warped_src_feat


class RefractiveCameraLayer(nn.Module):
    """
    Flat-port refractive camera model.

    The predicted depth is an underwater ray distance:
        X_ref = Q2 + depth * dir_water
    where Q2 is the outer glass interface point in the reference camera frame.
    """
    def __init__(self, z_inner=10.0, glass_thickness=5.0, n_air=1.0,
                 n_glass=1.52, n_water=1.333, learnable=True,
                 newton_iters=4, depth_chunk=4):
        super().__init__()
        self.n_air = float(n_air)
        self.n_water = float(n_water)
        self.newton_iters = int(newton_iters)
        self.depth_chunk = int(depth_chunk)
        if learnable:
            self.z_inner = nn.Parameter(torch.tensor(float(z_inner)))
            self.glass_thickness = nn.Parameter(torch.tensor(float(glass_thickness)))
            self.n_glass = nn.Parameter(torch.tensor(float(n_glass)))
        else:
            self.register_buffer("z_inner", torch.tensor(float(z_inner)))
            self.register_buffer("glass_thickness", torch.tensor(float(glass_thickness)))
            self.register_buffer("n_glass", torch.tensor(float(n_glass)))
        self._grid_cache = {}
        self._eval_ray_cache = {}

    def _pixel_grid(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self._grid_cache:
            grid = create_meshgrid(H, W, normalized_coordinates=False,
                                   device=device).to(dtype)
            grid = rearrange(grid, '1 h w c -> 1 h w c')
            self._grid_cache[key] = grid
        return self._grid_cache[key]

    def _cache_key(self, K, H, W):
        if self.training or K.requires_grad or K.shape[0] != 1:
            return None
        k0 = tuple(K[0].detach().float().cpu().reshape(-1).tolist())
        return (H, W, str(K.device), K.dtype, k0,
                float(self.z_inner.detach().cpu()),
                float(self.glass_thickness.detach().cpu()),
                float(self.n_glass.detach().cpu()))

    def refractive_rays(self, K, H, W):
        """
        K: (B, 3, 3)
        returns Q2, dir_water: (B, H, W, 3)
        """
        B = K.shape[0]
        device, dtype = K.device, K.dtype
        key = self._cache_key(K, H, W)
        if key is not None and key in self._eval_ray_cache:
            Q2, dir_water = self._eval_ray_cache[key]
            return Q2.expand(B, -1, -1, -1), dir_water.expand(B, -1, -1, -1)

        grid = self._pixel_grid(H, W, device, dtype)
        x = grid[..., 0].expand(B, -1, -1)
        y = grid[..., 1].expand(B, -1, -1)
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        x = (x - cx) / fx
        y = (y - cy) / fy
        dir_air = F.normalize(torch.stack([x, y, torch.ones_like(x)], dim=-1), dim=-1)
        radial_xy = dir_air[..., :2]
        radial_norm = torch.sqrt(torch.sum(radial_xy ** 2, dim=-1, keepdim=True))
        radial = radial_xy / radial_norm.clamp_min(1e-8)
        radial = torch.where(radial_norm > 1e-8, radial, torch.zeros_like(radial))

        theta_air = torch.atan2(radial_norm.squeeze(-1), dir_air[..., 2])
        sin_glass = (self.n_air / self.n_glass) * torch.sin(theta_air)
        theta_glass = torch.asin(sin_glass.clamp(-0.999999, 0.999999))
        dir_glass = torch.cat([radial * torch.sin(theta_glass)[..., None],
                               torch.cos(theta_glass)[..., None]], dim=-1)

        t1 = self.z_inner / dir_air[..., 2].clamp_min(1e-8)
        Q1 = dir_air * t1[..., None]
        t2 = self.glass_thickness / dir_glass[..., 2].clamp_min(1e-8)
        Q2 = Q1 + dir_glass * t2[..., None]

        sin_water = (self.n_glass / self.n_water) * torch.sin(theta_glass)
        theta_water = torch.asin(sin_water.clamp(-0.999999, 0.999999))
        dir_water = torch.cat([radial * torch.sin(theta_water)[..., None],
                               torch.cos(theta_water)[..., None]], dim=-1)
        dir_water = F.normalize(dir_water, dim=-1)

        if key is not None:
            self._eval_ray_cache[key] = (Q2[:1], dir_water[:1])
        return Q2, dir_water

    def backproject(self, depth_values, K_ref):
        B, D, H, W = depth_values.shape
        Q2, dir_water = self.refractive_rays(K_ref, H, W)
        Q2 = Q2[:, None].expand(B, D, H, W, 3)
        dir_water = dir_water[:, None].expand(B, D, H, W, 3)
        return Q2 + depth_values[..., None] * dir_water

    def z_to_ray_depth(self, z_depth, K_ref):
        """
        Convert camera-coordinate z depth to underwater ray distance.
        z_depth can be (B, D, H, W) or (B, 1, H, W).
        """
        B, _, H, W = z_depth.shape
        Q2, dir_water = self.refractive_rays(K_ref, H, W)
        q2_z = Q2[..., 2].view(B, 1, H, W)
        dir_z = dir_water[..., 2].view(B, 1, H, W).clamp_min(1e-8)
        return torch.clamp_min((z_depth - q2_z) / dir_z, 1e-7)

    def ray_to_z_depth(self, ray_depth, K_ref):
        """
        Convert underwater ray distance to camera-coordinate z depth.
        ray_depth can be (B, D, H, W) or (B, H, W).
        """
        has_depth_dim = ray_depth.dim() == 4
        if has_depth_dim:
            B, _, H, W = ray_depth.shape
        else:
            B, H, W = ray_depth.shape
        Q2, dir_water = self.refractive_rays(K_ref, H, W)
        q2_z = Q2[..., 2]
        dir_z = dir_water[..., 2]
        if has_depth_dim:
            q2_z = q2_z[:, None]
            dir_z = dir_z[:, None]
        return q2_z + ray_depth * dir_z

    def project(self, xyz, K_src):
        B = xyz.shape[0]
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        fx = K_src[:, 0, 0].view(B, 1, 1, 1)
        fy = K_src[:, 1, 1].view(B, 1, 1, 1)
        cx = K_src[:, 0, 2].view(B, 1, 1, 1)
        cy = K_src[:, 1, 2].view(B, 1, 1, 1)

        r = torch.sqrt(X ** 2 + Y ** 2 + 1e-12)
        z1 = self.z_inner
        tg = self.glass_thickness
        zw = Z - z1 - tg
        s = self.n_air * r / torch.sqrt(r ** 2 + Z ** 2 + 1e-12)
        max_s = min(self.n_air, self.n_water) - 1e-6
        s = torch.clamp(s, 1e-6, max_s)

        for _ in range(self.newton_iters):
            air_term = (self.n_air ** 2 - s ** 2).clamp_min(1e-8)
            glass_term = (self.n_glass ** 2 - s ** 2).clamp_min(1e-8)
            water_term = (self.n_water ** 2 - s ** 2).clamp_min(1e-8)
            A = z1 / torch.sqrt(air_term)
            B_glass = tg / torch.sqrt(glass_term)
            C = zw / torch.sqrt(water_term)
            f = s * (A + B_glass + C) - r
            dA = z1 * s / air_term.pow(1.5)
            dB = tg * s / glass_term.pow(1.5)
            dC = zw * s / water_term.pow(1.5)
            df = A + B_glass + C + s * (dA + dB + dC)
            s = torch.clamp(s - f / (df + 1e-8), 1e-6, max_s)

        tan_theta_air = torch.tan(torch.asin((s / self.n_air).clamp(-0.999999, 0.999999)))
        ux = X / r.clamp_min(1e-8)
        uy = Y / r.clamp_min(1e-8)
        center = r <= 1e-8
        u = fx * ux * tan_theta_air + cx
        v = fy * uy * tan_theta_air + cy
        u = torch.where(center, cx.expand_as(u), u)
        v = torch.where(center, cy.expand_as(v), v)
        return torch.stack([u, v], dim=-1)


def refractive_warp(src_feat, rel_pose, K_ref, K_src, depth_values,
                    refractive_camera):
    """
    src_feat: (B, C, H, W)
    rel_pose: (B, 4, 4), transforms reference camera coordinates to source camera coordinates
    K_ref, K_src: (B, 3, 3), scaled to the feature level
    depth_values: (B, D, H, W), underwater ray distance
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    depth_chunk = max(1, getattr(refractive_camera, "depth_chunk", D))
    warped_chunks = []
    for _, _, warped_src_feat in refractive_warp_chunks(
            src_feat, rel_pose, K_ref, K_src, depth_values,
            refractive_camera, depth_chunk):
        warped_chunks += [warped_src_feat]
    return torch.cat(warped_chunks, dim=2)


def refractive_projection_grid(rel_pose, K_ref, K_src, depth_values,
                               refractive_camera, H, W):
    xyz_ref = refractive_camera.backproject(depth_values, K_ref)
    xyz_ref_h = torch.cat([xyz_ref, torch.ones_like(xyz_ref[..., :1])], dim=-1)
    xyz_src = torch.matmul(rel_pose[:, None, None, None, :3, :],
                           xyz_ref_h[..., None]).squeeze(-1)
    src_grid = refractive_camera.project(xyz_src, K_src)
    valid_depth = xyz_src[..., 2] > 1e-7
    src_x = torch.where(valid_depth, src_grid[..., 0], src_grid.new_full((), W))
    src_y = torch.where(valid_depth, src_grid[..., 1], src_grid.new_full((), H))
    src_x = src_x / ((W - 1) / 2) - 1
    src_y = src_y / ((H - 1) / 2) - 1
    src_grid = torch.stack([src_x, src_y], dim=-1)
    return rearrange(src_grid, 'b d h w c -> b d (h w) c')


def refractive_warp_chunks(src_feat, rel_pose, K_ref, K_src, depth_values,
                           refractive_camera, depth_chunk=None):
    """
    Yield warped feature chunks along the depth dimension to reduce peak memory.
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    if depth_chunk is None:
        depth_chunk = getattr(refractive_camera, "depth_chunk", D)
    depth_chunk = max(1, int(depth_chunk))
    refraction_params = [refractive_camera.z_inner,
                         refractive_camera.glass_thickness,
                         refractive_camera.n_glass]
    needs_grid_grad = torch.is_grad_enabled() and any(
        getattr(param, "requires_grad", False) for param in refraction_params)
    for start in range(0, D, depth_chunk):
        end = min(start + depth_chunk, D)
        depth_chunk_values = depth_values[:, start:end]
        if needs_grid_grad:
            src_grid = refractive_projection_grid(rel_pose, K_ref, K_src,
                                                  depth_chunk_values,
                                                  refractive_camera, H, W)
        else:
            with torch.no_grad():
                src_grid = refractive_projection_grid(rel_pose, K_ref, K_src,
                                                      depth_chunk_values,
                                                      refractive_camera, H, W)
        warped_src_feat = F.grid_sample(src_feat, src_grid, mode='bilinear',
                                        padding_mode='zeros', align_corners=True)
        warped_src_feat = rearrange(warped_src_feat, 'b c d (h w) -> b c d h w',
                                    h=H, w=W)
        yield start, end, warped_src_feat


def depth_regression(p, depth_values):
    """
    p: probability volume (B, D, H, W)
    depth_values: discrete depth values (B, D, H, W) or (D)
    inverse: depth_values is inverse depth or not
    """
    if depth_values.dim() == 1:
        depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
    depth = reduce(p*depth_values, 'b d h w -> b h w', 'sum').to(depth_values.dtype)
    return depth
