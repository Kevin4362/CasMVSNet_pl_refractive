from einops import reduce, rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .modules import *

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential( 
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, 
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x) # (B, 8, H, W)
        conv1 = self.conv1(conv0) # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1) # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2) # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1) # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0) # (B, 8, H, W)

        feats = {"level_0": feat0,
                 "level_1": feat1,
                 "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x


class CascadeMVSNet(nn.Module):
    def __init__(self, n_depths=[8, 32, 48],
                       interval_ratios=[1, 2, 4],
                       num_groups=1,
                       norm_act=InPlaceABN,
                       use_refractive=False,
                       refractive_params=None,
                       cost_reg_checkpoint=False):
        super(CascadeMVSNet, self).__init__()
        self.levels = 3 # 3 depth levels
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.G = num_groups # number of groups in groupwise correlation
        self.use_refractive = use_refractive
        self.cost_reg_checkpoint = cost_reg_checkpoint
        self.feature = FeatureNet(norm_act)
        if self.use_refractive:
            refractive_params = refractive_params or {}
            self.refractive_camera = RefractiveCameraLayer(**refractive_params)
        for l in range(self.levels):
            if self.G > 1:
                cost_reg_l = CostRegNet(self.G, norm_act)
            else:
                cost_reg_l = CostRegNet(8*2**l, norm_act)
            setattr(self, f'cost_reg_{l}', cost_reg_l)

    def predict_depth(self, feats, proj_mats, depth_values, cost_reg,
                      refractive_mats=None, depth_values_z=None):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V-1, 3, 4)
        # depth_values: (B, D, H, W), pinhole depth or refractive ray distance
        # depth_values_z: optional (B, D, H, W), z-depth values for refractive output/loss
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = rearrange(src_feats, 'b vm1 c h w -> vm1 b c h w') # (V-1, B, C, h, w)
        proj_mats = rearrange(proj_mats, 'b vm1 x y -> vm1 b x y') # (V-1, B, 3, 4)
        if self.use_refractive:
            if refractive_mats is None:
                raise ValueError('refractive_mats is required when use_refractive=True')
            K_ref = refractive_mats['K_ref']
            K_srcs = rearrange(refractive_mats['K_srcs'], 'b vm1 x y -> vm1 b x y')
            rel_poses = rearrange(refractive_mats['rel_poses'], 'b vm1 x y -> vm1 b x y')

        ref_volume = rearrange(ref_feats, 'b c h w -> b c 1 h w')
        if self.G == 1:
            if self.use_refractive:
                ref_volume_base = ref_volume
                volume_sum = None
                volume_sq_sum = None
            else:
                ref_volume = repeat(ref_volume, 'b c 1 h w -> b c d h w', d=D) # (B, C, D, h, w)
                volume_sum = ref_volume
                volume_sq_sum = ref_volume ** 2
        else:
            ref_volume = repeat(ref_volume, 'b c 1 h w -> b c d h w', d=D) # (B, C, D, h, w)
            ref_volume = ref_volume.view(B, self.G, C//self.G, *ref_volume.shape[-3:])
            volume_sum = 0
        del ref_feats

        volume_variance = None
        if self.use_refractive and self.G == 1:
            depth_chunk = max(1, getattr(self.refractive_camera, "depth_chunk", D))
            chunk_ranges = [(s, min(s + depth_chunk, D))
                            for s in range(0, D, depth_chunk)]
            volume_sum_chunks = [
                ref_volume_base.expand(-1, -1, e-s, -1, -1)
                for s, e in chunk_ranges
            ]
            volume_sq_sum_chunks = [chunk * chunk for chunk in volume_sum_chunks]
            for i, src_feat in enumerate(src_feats):
                for ci, (s, e, warped_chunk) in enumerate(
                        refractive_warp_chunks(src_feat, rel_poses[i], K_ref,
                                               K_srcs[i], depth_values,
                                               self.refractive_camera, depth_chunk)):
                    warped_chunk = warped_chunk.to(ref_volume_base.dtype)
                    volume_sum_chunks[ci] = volume_sum_chunks[ci] + warped_chunk
                    volume_sq_sum_chunks[ci] = (
                        volume_sq_sum_chunks[ci] + warped_chunk * warped_chunk)
                    del warped_chunk
                del src_feat
            volume_variance_chunks = []
            for volume_sum_chunk, volume_sq_sum_chunk in zip(volume_sum_chunks,
                                                             volume_sq_sum_chunks):
                volume_mean_chunk = volume_sum_chunk / V
                volume_variance_chunk = volume_sq_sum_chunk.div(V).sub_(
                    volume_mean_chunk.pow(2))
                volume_variance_chunks += [volume_variance_chunk]
                del volume_sum_chunk, volume_sq_sum_chunk, volume_mean_chunk
            volume_variance = torch.cat(volume_variance_chunks, dim=2)
            del volume_variance_chunks, ref_volume_base
            del volume_sum_chunks, volume_sq_sum_chunks
        else:
            for i, (src_feat, proj_mat) in enumerate(zip(src_feats, proj_mats)):
                if self.use_refractive:
                    warped_volume = refractive_warp(src_feat, rel_poses[i], K_ref,
                                                    K_srcs[i], depth_values,
                                                    self.refractive_camera)
                else:
                    warped_volume = homo_warp(src_feat, proj_mat, depth_values)
                warped_volume = warped_volume.to(ref_volume.dtype)
                if self.G == 1:
                    if self.training:
                        volume_sum = volume_sum + warped_volume
                        volume_sq_sum = volume_sq_sum + warped_volume ** 2
                    else:
                        volume_sum += warped_volume
                        volume_sq_sum += warped_volume.pow_(2)
                else:
                    warped_volume = warped_volume.view_as(ref_volume)
                    if self.training:
                        volume_sum = volume_sum + warped_volume # (B, G, C//G, D, h, w)
                    else:
                        volume_sum += warped_volume
                del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats
        # aggregate multiple feature volumes by variance
        if volume_variance is not None:
            pass
        elif self.G == 1:
            volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
            del volume_sq_sum, volume_sum
        else:
            volume_variance = reduce(volume_sum*ref_volume,
                                     'b g c d h w -> b g d h w', 'mean').div_(V-1) # (B, G, D, h, w)
            del volume_sum, ref_volume
        
        if self.cost_reg_checkpoint and self.training:
            cost_reg = checkpoint(cost_reg, volume_variance)
        else:
            cost_reg = cost_reg(volume_variance)
        cost_reg = rearrange(cost_reg, 'b 1 d h w -> b d h w')
        prob_volume = F.softmax(cost_reg, 1) # (B, D, h, w)
        del cost_reg
        depth = depth_regression(prob_volume, depth_values)
        depth_ray = depth
        if self.use_refractive:
            if depth_values_z is None:
                raise ValueError('depth_values_z is required when use_refractive=True')
            depth = depth_regression(prob_volume, depth_values_z)
        
        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                      pad=(0, 0, 0, 0, 1, 2)),
                                                (4, 1, 1), stride=1).squeeze(1) # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(prob_volume,
                                           torch.arange(D,
                                                        device=prob_volume.device,
                                                        dtype=prob_volume.dtype)
                                          ).long() # (B, h, w)
            depth_index = torch.clamp(depth_index, 0, D-1)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(prob_volume_sum4, 1, 
                                      depth_index.unsqueeze(1)).squeeze(1) # (B, h, w)

        return depth, depth_ray, confidence

    def forward(self, imgs, proj_mats, init_depth_min, depth_interval,
                camera_mats=None):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
        # camera_mats: optional dict with intrinsics (B,V,L,3,3) and rel_poses (B,V-1,L,4,4)
        # init_depth_min, depth_interval: (B) or float
        B, V, _, H, W = imgs.shape
        results = {}

        imgs = imgs.reshape(B*V, 3, H, W)
        feats = self.feature(imgs) # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        
        for l in reversed(range(self.levels)): # (2, 1, 0)
            feats_l = feats[f"level_{l}"] # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:]) # (B, V, C, h, w)
            proj_mats_l = proj_mats[:, :, l] # (B, V-1, 3, 4)
            refractive_mats_l = None
            if self.use_refractive:
                if camera_mats is None:
                    raise ValueError('camera_mats is required when use_refractive=True')
                intrinsics_l = camera_mats['intrinsics'][:, :, l]
                refractive_mats_l = {
                    'K_ref': intrinsics_l[:, 0],
                    'K_srcs': intrinsics_l[:, 1:],
                    'rel_poses': camera_mats['rel_poses'][:, :, l],
                }
            depth_interval_l = depth_interval * self.interval_ratios[l]
            D = self.n_depths[l]
            if l == self.levels-1: # coarsest level
                h, w = feats_l.shape[-2:]
                if isinstance(init_depth_min, float):
                    depth_values = init_depth_min + depth_interval_l * \
                                   torch.arange(0, D,
                                                device=imgs.device,
                                                dtype=imgs.dtype) # (D)
                    depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
                    depth_values = repeat(depth_values, '1 d 1 1 -> b d h w', b=B, h=h, w=w)
                else:
                    depth_values = init_depth_min + depth_interval_l * \
                                   rearrange(torch.arange(0, D,
                                                          device=imgs.device,
                                                          dtype=imgs.dtype),
                                             'd -> 1 d') # (B, D)
                    depth_values = rearrange(depth_values, 'b d -> b d 1 1')
                    depth_values = repeat(depth_values, 'b d 1 1 -> b d h w', h=h, w=w)
            else:
                depth_lm1 = depth_l.detach() # the depth of previous level
                depth_lm1 = F.interpolate(rearrange(depth_lm1, 'b h w -> b 1 h w'),
                                          scale_factor=2, mode='bilinear',
                                          align_corners=True) # (B, 1, h, w)
                depth_values = get_depth_values(depth_lm1, D, depth_interval_l)
                del depth_lm1
            depth_values_z = None
            if self.use_refractive:
                depth_values_z = depth_values
                depth_values = self.refractive_camera.z_to_ray_depth(
                    depth_values_z, refractive_mats_l['K_ref'])
            depth_l, depth_ray_l, confidence_l = self.predict_depth(
                feats_l, proj_mats_l, depth_values,
                getattr(self, f'cost_reg_{l}'), refractive_mats_l,
                depth_values_z)
            del feats_l, proj_mats_l, depth_values
            results[f"depth_{l}"] = depth_l
            if self.use_refractive:
                results[f"depth_ray_{l}"] = depth_ray_l
            results[f"confidence_{l}"] = confidence_l
            

        return results
class RefractiveCamera(nn.Module):

    def __init__(
            self,
            H,
            W,
            K,
            z_inner=10.0,
            glass_thickness=5.0,
            n_air=1.0,
            n_glass=1.52,
            n_water=1.333,
            learnable=True,
    ):
        super().__init__()

        self.H = H
        self.W = W

        self.register_buffer(
            "K",
            torch.tensor(K, dtype=torch.float32)
        )

        self.n_air = n_air
        self.n_water = n_water

        if learnable:
            self.z_inner = nn.Parameter(
                torch.tensor(float(z_inner))
            )

            self.glass_thickness = nn.Parameter(
                torch.tensor(float(glass_thickness))
            )

            self.n_glass = nn.Parameter(
                torch.tensor(float(n_glass))
            )
        else:
            self.register_buffer(
                "z_inner",
                torch.tensor(float(z_inner))
            )

            self.register_buffer(
                "glass_thickness",
                torch.tensor(float(glass_thickness))
            )

            self.register_buffer(
                "n_glass",
                torch.tensor(float(n_glass))
            )

        self._build_pixel_grid()

    def _build_pixel_grid(self):

        ys, xs = torch.meshgrid(
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij"
        )

        self.register_buffer(
            "pixel_x",
            xs.float()
        )

        self.register_buffer(
            "pixel_y",
            ys.float()
        )

    def _snell_theta(self, s, n):

        s = torch.clamp(
            s / n,
            -0.999999,
            0.999999
        )

        return torch.asin(s)

    def backproject(self, depth):

        device = depth.device

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        x = (self.pixel_x - cx) / fx
        y = (self.pixel_y - cy) / fy

        dir_air = torch.stack(
            [x, y, torch.ones_like(x)],
            dim=-1
        )

        dir_air = F.normalize(
            dir_air,
            dim=-1
        )

        # ---------- Q1 ----------

        t1 = self.z_inner / dir_air[..., 2]

        Q1 = (
                dir_air *
                t1.unsqueeze(-1)
        )

        # ---------- angle ----------

        r_air = torch.sqrt(
            dir_air[..., 0] ** 2 +
            dir_air[..., 1] ** 2
        )

        theta_air = torch.atan2(
            r_air,
            dir_air[..., 2]
        )

        theta_glass = torch.asin(
            self.n_air /
            self.n_glass *
            torch.sin(theta_air)
        )

        radial = torch.stack(
            [
                dir_air[..., 0],
                dir_air[..., 1]
            ],
            dim=-1
        )

        radial = F.normalize(
            radial,
            dim=-1
        )

        # ---------- glass ray ----------

        dir_glass = torch.cat(
            [
                radial *
                torch.sin(theta_glass)[..., None],

                torch.cos(theta_glass)[..., None]
            ],
            dim=-1
        )

        t2 = (
                self.glass_thickness /
                dir_glass[..., 2]
        )

        Q2 = (
                Q1 +
                dir_glass * t2.unsqueeze(-1)
        )

        # ---------- water ray ----------

        theta_water = torch.asin(
            self.n_glass /
            self.n_water *
            torch.sin(theta_glass)
        )

        dir_water = torch.cat(
            [
                radial *
                torch.sin(theta_water)[..., None],

                torch.cos(theta_water)[..., None]
            ],
            dim=-1
        )

        dir_water = F.normalize(
            dir_water,
            dim=-1
        )

        # ---------- depth hypotheses ----------

        B, D, H, W = depth.shape

        Q2 = (
            Q2[None, None]
            .expand(B, D, H, W, 3)
        )

        dir_water = (
            dir_water[None, None]
            .expand(B, D, H, W, 3)
        )

        xyz = (
                Q2 +
                depth[..., None] *
                dir_water
        )

        return xyz

    def project(
            self,
            xyz,
            newton_iters=4
    ):
        """
        xyz:
        [...,3]

        return:
        [...,2]
        """

        X = xyz[..., 0]
        Y = xyz[..., 1]
        Z = xyz[..., 2]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        r = torch.sqrt(
            X ** 2 + Y ** 2 + 1e-12
        )

        z1 = self.z_inner

        tg = self.glass_thickness

        zw = (
                Z -
                z1 -
                tg
        )

        # 初值
        s = (
                self.n_air *
                r /
                torch.sqrt(
                    r ** 2 + Z ** 2
                )
        )

        s = torch.clamp(
            s,
            1e-6,
            self.n_air - 1e-6
        )

        # Newton

        for _ in range(newton_iters):
            A = (
                    z1 /
                    torch.sqrt(
                        self.n_air ** 2 - s ** 2
                    )
            )

            B = (
                    tg /
                    torch.sqrt(
                        self.n_glass ** 2 - s ** 2
                    )
            )

            C = (
                    zw /
                    torch.sqrt(
                        self.n_water ** 2 - s ** 2
                    )
            )

            f = s * (A + B + C) - r

            dA = (
                    z1 * s /
                    (self.n_air ** 2 - s ** 2) ** 1.5
            )

            dB = (
                    tg * s /
                    (self.n_glass ** 2 - s ** 2) ** 1.5
            )

            dC = (
                    zw * s /
                    (self.n_water ** 2 - s ** 2) ** 1.5
            )

            df = (
                    A + B + C +
                    s * (dA + dB + dC)
            )

            s = s - f / (df + 1e-8)

        theta_air = torch.asin(
            s / self.n_air
        )

        tan_theta = torch.tan(
            theta_air
        )

        ux = X / r
        uy = Y / r

        u = (
                fx *
                ux *
                tan_theta +
                cx
        )

        v = (
                fy *
                uy *
                tan_theta +
                cy
        )

        return torch.stack(
            [u, v],
            dim=-1
        )
