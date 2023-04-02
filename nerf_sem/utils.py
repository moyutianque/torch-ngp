import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
import h5py
from constants import d3_40_colors_rgb
from math import log10, sqrt

from nerf_sem.reprojection_loss import reprojection_loss, test_reprojection

def np_PSNR(original, compressed):
    if original.shape[2] == 4:
        original = original[:,:,:3]
    
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def np_diff_sem(original, synthesis):
    msk = original == synthesis
    ratio_diff = np.mean(msk)
    
    return ratio_diff

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}
    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2] zehao@Dec 7 all point coord of the patch
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten index

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    # NOTE zehao @ Dec 7 pixel coord to cam coord
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3) # NOTE cam coord to world coord

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o # zehao ray origin
    results['rays_d'] = rays_d # zehao ray direction

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u

def extract_fields_sem(bound_min, bound_max, resolution, query_func, S=128, sem_map_type='id', return_pts=False):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u_density = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    if return_pts:
        u_pts = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)

    if sem_map_type == 'rgb':
        u_sem = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    else:
        u_sem = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    density, sem_out = query_func(pts)
                    val = density.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    if return_pts:
                        pt_reshape = pts.reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy() # [S, 3] --> [x, y, z, 3]
                        u_pts[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs), :] = pt_reshape

                    u_density[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val

                    if sem_map_type == 'rgb':
                        sem_val = sem_out.reshape(len(xs), len(ys), len(zs), -1).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                        u_sem[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs), :] = sem_val
                    else:
                        sem_out = torch.argmax(sem_out, dim=-1)
                        sem_val = sem_out.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                        u_sem[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = sem_val
    if return_pts:
        return u_density, u_sem, u_pts
    return u_density, u_sem


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def depth_reg(depth_tensor):
    """ [B, num_patch, patch_size, patch_size] """
    i_mse = nn.functional.mse_loss(depth_tensor[:,:-1,:], depth_tensor[:,1:,:])
    j_mse = nn.functional.mse_loss(depth_tensor[:,:,:-1], depth_tensor[:,:,1:]) 
    
    ds_loss = i_mse + j_mse
    return ds_loss

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 criterion_sem=None,
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 save_iter = 100,
                 vae=None
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.vae = vae

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        if isinstance(criterion_sem, nn.Module):
            criterion_sem.to(self.device)
        
        self.criterion = criterion
        self.criterion_sem = criterion_sem

        # optionally use LPIPS loss for patch-based training
        # if self.opt.patch_size > 1:
        #     import lpips
        #     self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.save_iter = save_iter
        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        
        if self.opt.latent:
            latents = data['latents']
            gt_latents = latents
            bg_color=1
            outputs = self.model.render_latent(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, global_step=self.global_step, **vars(self.opt))
            
            loss = {}
            pred_latent = outputs['latent']
            if pred_latent is not None:
                pred_latent = self.model.latent_proj(pred_latent)
                
                if self.opt.patch_size > 1: 
                    gt_latents = gt_latents.view(-1, self.opt.patch_size, self.opt.patch_size, 4).permute(0, 3, 1, 2).contiguous()
                    pred_latent = pred_latent.view(-1, self.opt.patch_size, self.opt.patch_size, 4).permute(0, 3, 1, 2).contiguous()

                    pred_rgb = self.vae.decode(pred_latent).sample
                    gt_rgb = self.vae.decode(gt_latents).sample
                    if os.environ.get('VIS_PATCH', False):
                        import matplotlib.pyplot as plt
                        for patch, patch_gt in zip(pred_rgb, gt_rgb):
                            im_p = patch.detach().cpu().permute(1,2,0).numpy().astype(float)
                            im_gt = patch_gt.detach().cpu().permute(1,2,0).numpy().astype(float)
                            plt.imshow(np.concatenate((np.clip(im_p, 0, 1), np.clip(im_gt, 0, 1)), axis=1))
                            plt.show()
                    # loss = self.criterion(pred_latent, gt_latents).mean((2,3))
                    loss = self.criterion(pred_rgb, gt_rgb).mean((1,2,3))
                else:
                    loss = self.criterion(pred_latent, gt_latents).mean(-1)
                loss = {
                    'loss_latent': loss.mean(),
                }

            if self.opt.low_res_img:
                gt_rgb = data['images']
                pred_rgb = outputs['image']
                loss_rgb = self.criterion(pred_rgb, gt_rgb).mean(-1)
                loss.update({"loss_rgb": loss_rgb.mean()})
            return pred_latent, gt_latents, loss

        else:
            images = data['images'] # [B, N, 3/4]
            if self.opt.depth_sup:
                gt_depth = data['images_depth']

            bg_color = 1
            gt_rgb = images

            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, global_step=self.global_step, **vars(self.opt))
            # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
            outputs_rd = None
            if 'rd_rays_o' in data:
                outputs_rd = self.model.render(data['rd_rays_o'], data['rd_rays_d'], staged=False, bg_color=torch.rand(3, device=self.device), perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, gloabl_step=self.global_step, **vars(self.opt))

            pred_rgb = outputs['image']
            if self.opt.depth_sup:
                pred_depth = outputs['depth']
            
            # NOTE: MSE depth loss
            loss_depth = None
            if self.opt.depth_sup:
                if not self.opt.radial_depth:
                    pred_depth = pred_depth * data['depth_radial2plane']
                loss_depth = torch.abs(torch.log(gt_depth)-torch.log(pred_depth))
                filtered_idx = (~torch.isinf(loss_depth)) & (~torch.isnan(loss_depth))
                loss_depth = loss_depth[filtered_idx].mean()
                print(loss_depth.item())
                # loss_depth = self.criterion(pred_depth, gt_depth).mean()

            # NOTE by zehao, default this is not called
            # # patch-based rendering
            loss_ds = None
            if self.opt.patch_size > 1: 
                gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
                pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

                pred_depth = outputs['depth'].view(-1, self.opt.patch_size, self.opt.patch_size)
                if self.opt.depth_reg:
                    loss_ds = depth_reg(pred_depth)
                    if outputs_rd is not None:
                        pred_depth2 = outputs_rd['depth'].view(-1, self.opt.patch_size, self.opt.patch_size)
                        loss_ds += depth_reg(pred_depth2)
                        
                if os.environ.get('VIS_PATCH', False):
                    import matplotlib.pyplot as plt
                    for patch, patch_gt in zip(pred_rgb, gt_rgb):
                        im_p = patch.detach().cpu().permute(1,2,0).numpy().astype(float)
                        im_gt = patch_gt.detach().cpu().permute(1,2,0).numpy().astype(float)
                        plt.imshow(np.concatenate((np.clip(im_p, 0, 1), np.clip(im_gt, 0, 1)), axis=1))
                        plt.show()
                loss = self.criterion(pred_rgb, gt_rgb).mean((1,2,3))
            else:
                # MSE loss
                loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

            loss_rgb = loss.mean()

            loss = {
                'loss_rgb': loss_rgb,
                "loss_depth": loss_depth,
                'loss_ds': loss_ds
            }

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, global_step=self.global_step,**vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        if self.opt.latent:
            outputs = self.model.render_latent(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, global_step=self.global_step, **vars(self.opt))
            pred_latent = outputs['latent'].reshape(-1, H, W, self.model.latent_dim)
            pred_latent = self.model.latent_proj(pred_latent)
            self.vae.eval()
            
            if self.opt.latent_space == 'img_size':
                pred_rgb = self.vae.decode(F.interpolate(pred_latent.permute(0, 3, 1, 2), scale_factor=(1/8, 1/8), mode='nearest')).sample
                pred_rgb = pred_rgb.permute(0,2,3,1)
                pred_rgb = torch.clip(pred_rgb, 0, 1).contiguous() # gui will failed if not contiguous
            else:
                pred_rgb = self.vae.decode(pred_latent.permute(0, 3, 1, 2)).sample
                pred_rgb = pred_rgb.permute(0,2,3,1)
                pred_rgb = torch.clip(pred_rgb, 0, 1).contiguous()

            if 'depth_normalized' in outputs:
                pred_depth_normalized = outputs['depth_normalized'].reshape(-1, H, W)
                pred_depth = outputs['depth'].reshape(-1, H, W)
            else:
                pred_depth_normalized = outputs['depth'].reshape(-1, H, W)
            return pred_rgb, pred_depth_normalized, None, pred_depth
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, global_step=self.global_step, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(-1, H, W, 3)
            pred_depth_normalized=None
            if 'depth_normalized' in outputs:
                pred_depth_normalized = outputs['depth_normalized'].reshape(-1, H, W)
                pred_depth = outputs['depth'].reshape(-1, H, W)
            else:
                pred_depth_normalized = outputs['depth'].reshape(-1, H, W)

            pred_sem = outputs['image_sem'].reshape(-1, H, W, self.model.sem_out_dim)

            return pred_rgb, pred_depth_normalized, pred_sem, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    def get_3dmap(self, resolution, sem_map_type, return_pts=False):
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    density_outputs = self.model.density(pts.to(self.device))
                    sem_out = self.model.sem(pts.to(self.device), **density_outputs)
            return density_outputs['sigma'], sem_out
        
        return extract_fields_sem(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, query_func=query_func, sem_map_type=sem_map_type, return_pts=return_pts)

    def save_3dmap(self, resolution=256, sem_map_type='rgb'):
        save_path = os.path.join(self.workspace, f'{self.name}_3dmap.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        density_out, sem_out = self.get_3dmap(resolution, sem_map_type, return_pts=False)

        with h5py.File(save_path, "w") as f:
            f.create_dataset("density", data=density_out)
            f.create_dataset("sem", data=sem_out)

    ### ------------------------------

    def train_old(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses_train, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
        
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_sem, preds_depth_ori = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
        
    def train(self, train_loader, step=16, val_data=None, sem_map_type='rgb'):
        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_low_res = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        self.error_map = train_loader._data.error_map
        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses_train, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss_dict = self.train_step(data)

            if 'loss_latent' in loss_dict:
                tot_loss = loss_dict['loss_latent'].clone() * 0.001
                if 'loss_rgb' in loss_dict:
                    tot_loss += loss_dict['loss_rgb']
                # tot_loss = loss_dict['loss_rgb'].clone()
            else: 
                tot_loss = loss_dict['loss_rgb'].clone()

            self.scaler.scale(tot_loss).backward()
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
 
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            if 'loss_latent' in loss_dict:
                total_loss += loss_dict['loss_latent'].detach()
            if 'loss_rgb' in loss_dict:
                total_low_res += loss_dict['loss_rgb'].detach()

            if self.global_step != 0 and (self.global_step % self.opt.save_iter == 0):
                self.get_view_sythesis(val_data, train_loader._data.intrinsics, train_loader._data.test_len, self.global_step, sem_map_type)
                self.model.training = True
                self.save_checkpoint(full=True, best=False, remove_old=False)
                self.epoch += 1

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step
        average_loss_low_res = total_low_res.item() / step

        outputs = {
            'l_latent': average_loss,
            'l_rgb': average_loss_low_res,
            'l_dist':0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs    


    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16, val_data=None, iters=0, sem_map_type='rgb'):
        if os.environ.get('GETDEPTH', False):
            # self.get_view_sythesis(val_data, train_loader._data.intrinsics, train_loader._data.test_len, self.global_step, sem_map_type)
            import ipdb;ipdb.set_trace() # breakpoint 849

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_loss_sem = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_loss_ds = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_loss_dist = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses_train, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss_dict = self.train_step(data)

            if 'loss_latent' in loss_dict:
                tot_loss = loss_dict['loss_latent'].clone()
            else: 
                tot_loss = loss_dict['loss_rgb'].clone()
            # tot_loss = 0
        
            # NOTE: loss merge
            # if (loss_dict['loss_sem'] is not None) and (self.global_step > self.opt.warmup_iter):
            #     sem_loss_w = 1. # semantic-nerf use 4e-2
            #     tot_loss += sem_loss_w * loss_dict['loss_sem']
            #     if (loss_dict['loss_dist'] is not None) and self.global_step > self.opt.dist_start:
            #         tot_loss += 1. * loss_dict['loss_dist']

            # if loss_dict['loss_depth'] is not None:
            #     print(loss_dict['loss_depth'].item())
            #     tot_loss += 0.01 * loss_dict['loss_depth']
                

            # if self.opt.post_3dmap_loss:
            #     if (not hasattr(self, 'pts_empty')) or (self.global_step % self.opt.density_sample_size == 1):
            #         from nerf_sem.semmap_post import map_filtering
            #         dm, sm, pts = self.get_3dmap(resolution=256, sem_map_type='id', return_pts=True)
            #         dm_out = map_filtering(dm, sm)
            #         pts = pts.reshape((-1,3))
            #         dm_out= dm_out.flatten()
            #         empty_idx = dm_out == 0
            #         self.pts_empty = pts[empty_idx]

            #     sample_idxs = random.sample([i for i in range(len(self.pts_empty))], min(1024, len(self.pts_empty)))
            #     pts_selected = self.pts_empty[sample_idxs]
            #     pts_selected = torch.from_numpy(pts_selected).to(self.device)

            #     # sample pts for mark
            #     density_outputs = self.model.density(pts_selected)
            #     # NOTE: with sem empty loss
            #     # sem_out = self.model.sem(pts_selected, **density_outputs) # float16 not support crossentropy
            #     # tot_loss += 0.001 * self.criterion_sem(sem_out.to(torch.float32), torch.zeros_like(density_outputs['sigma']).to(torch.float32))

            #     raw_loss = self.criterion(density_outputs['sigma'], torch.zeros_like(density_outputs['sigma']))
            #     filtered_idx = ~torch.isinf(raw_loss)
            #     empty_density_loss = 0.0001 * raw_loss[filtered_idx].mean()
            #     print(empty_density_loss.item(), torch.sum(filtered_idx))
            #     tot_loss += empty_density_loss
            self.scaler.scale(tot_loss).backward()
                
            self.scaler.step(self.optimizer)
            self.scaler.update()

            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            # latent_loss = loss_dict['loss_latent'].detach()
            total_loss += loss_dict['loss_latent'].detach()
            # if loss_dict['loss_sem'] is not None:
            #     total_loss_sem += loss_dict['loss_sem'].detach()
            
            # if loss_dict['loss_dist'] is not None:
            #     total_loss_dist += loss_dict['loss_dist'].detach()
            

            if self.global_step != 0 and (self.global_step % self.opt.save_iter == 0):
                self.get_view_sythesis(val_data, train_loader._data.intrinsics, train_loader._data.test_len, self.global_step, sem_map_type)
                self.model.training = True
                self.save_checkpoint(full=True, best=False, remove_old=False)
                self.epoch += 1

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step
        # average_loss_sem = total_loss_sem.item() / step
        # average_loss_ds = total_loss_ds.item() / step
        # average_loss_dist = total_loss_dist.item() / step

        # if not self.scheduler_update_every_step:
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         self.lr_scheduler.step(average_loss)
        #     else:
        #         self.lr_scheduler.step()

        outputs = {
            'l_rgb': average_loss,
            'l_sem': 0,
            'l_dist':0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    def get_view_sythesis(self, val_data, intrinsics, test_len, iters, sem_map_type='rgb'):
        val_results = []
        save_path = os.path.join(self.workspace, 'vis_eval')
        os.makedirs(save_path, exist_ok=True)
        psnr_list = []
        psnr_list_oldview = []
        for j, item in enumerate(zip(*val_data)):
            (p, im, im_sem) = item
            test_output = self.test_gui(
                    p, intrinsics, 
                    im.shape[1], im.shape[0], bg_color=torch.ones(3).float()
                )

            img_rgb = (im * 255).astype(np.uint8)
            pred = (test_output['image'] * 255).astype(np.uint8)
            pred_depth = (test_output['depth'] * 255).astype(np.uint8)
            depth_ori = test_output['depth_ori']
            
            prefix = ''
            if len(val_data[0])-j <= test_len:
                prefix = 'newview'
                # psnr_list.append(np_PSNR(img_rgb, pred))
            else:
                # psnr_list_oldview.append(np_PSNR(img_rgb, pred))
                pass

            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}gt.png'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_depth_{prefix}.png'), pred_depth)
            val_results.append(test_output)

        self.model.train()

    def get_view_sythesis_full(self, val_data, intrinsics, test_len, iters, sem_map_type='rgb'):
        val_results = []
        save_path = os.path.join(self.workspace, 'vis_eval')
        os.makedirs(save_path, exist_ok=True)
        psnr_list = []
        psnr_list_oldview = []
        sem_diff = []
        sem_diff_old = []
        for j, item in enumerate(zip(*val_data)):
            if self.opt.depth_sup:
                if self.opt.reprojection_loss:
                    (p, im, im_sem, im_depth, nearby_view) = item
                else:
                    (p, im, im_sem, im_depth) = item
            else:
                (p, im, im_sem) = item
            
            # NOTE ground truth reprojection verification
            if self.opt.reprojection_loss:
                H, W = im.shape[:2]
                xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                xs = xs.flatten(); ys = ys.flatten()
                pix_pts = np.transpose(np.vstack([xs, ys]))
 
                projected_rgb = reprojection_loss(pix_pts, im_depth, intrinsics, p, nearby_view['pose'], nearby_view['depth'], rgb = im)
                
                img_rgb = (projected_rgb * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_wrap.png'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                target_rgb = (nearby_view['rgb'] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_target.png'), cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
                ori_rgb = (im* 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_original.png'), cv2.cvtColor(ori_rgb, cv2.COLOR_RGB2BGR))
                import ipdb;ipdb.set_trace() # breakpoint 1024

            test_output = self.test_gui(
                    p, intrinsics, 
                    im.shape[1], im.shape[0], bg_color=torch.ones(3, dtype=torch.float32)
                )

            img_rgb = (im * 255).astype(np.uint8)
            pred = (test_output['image'] * 255).astype(np.uint8)
            pred_depth = (test_output['depth'] * 255).astype(np.uint8)
            depth_ori = test_output['depth_ori']

            if sem_map_type == 'rgb':
                img_sem = (im_sem * 255).astype(np.uint8)
                pred_sem_raw = test_output['sem']
                pred_sem = (pred_sem_raw * 255).astype(np.uint8)
            else:
                msk_zero = im_sem == 0
                img_sem = d3_40_colors_rgb[im_sem % 40 + 1].astype(np.uint8)
                img_sem[msk_zero] = 0

                pred_sem_raw = np.argmax(test_output['sem'], axis=-1).astype(int)
                msk_zero = pred_sem_raw == 0
                pred_sem = d3_40_colors_rgb[pred_sem_raw % 40 + 1].astype(np.uint8)
                pred_sem[msk_zero] = 0

            prefix = ''
            if len(val_data[0])-j <= test_len:
                prefix = 'newview'
                psnr_list.append(np_PSNR(img_rgb, pred))
                if sem_map_type == 'rgb':
                    sem_diff.append(np_PSNR(img_sem, pred_sem))
                else:
                    sem_diff.append(np_diff_sem(im_sem, pred_sem_raw))
            else:
                psnr_list_oldview.append(np_PSNR(img_rgb, pred))
                if sem_map_type == 'rgb':
                    sem_diff_old.append(np_PSNR(img_sem, pred_sem))
                else:
                    sem_diff_old.append(np_diff_sem(im_sem, pred_sem_raw))

            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}gt.png'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_sem_{prefix}gt.png'), cv2.cvtColor(img_sem, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_sem_{prefix}.png'), cv2.cvtColor(pred_sem, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_depth_{prefix}.png'), pred_depth)

            f_path = os.path.join(save_path, f'{iters}_{j}_depth_{prefix}.npy')
            np.save(f_path, depth_ori)
            if self.opt.depth_sup:
                f_path = os.path.join(save_path, f'{iters}_{j}_depth_{prefix}_gt.npy')
                np.save(f_path, im_depth)

            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_depth_{prefix}.png'), pred_depth)
            val_results.append(test_output)
        
        with open(os.path.join(save_path, 'psnr_records.txt'), 'a') as file:
            file.write(f"PSNR results for iter {iters}\n")
            file.write("[" + ' '.join(str(v) for v in psnr_list_oldview) + "]" + "\n")
            file.write("Old view PSNR avg: "+ str(np.mean(psnr_list_oldview)) + "]" + "\n")
            file.write("[" + ' '.join(str(v) for v in psnr_list) + "\n")
            file.write("Novel view PSNR avg: "+ str(np.mean(psnr_list))+ "\n\n")

            file.write(f"Sem diff results for iter {iters}\n")
            file.write("[" + ' '.join(str(v) for v in sem_diff_old) + "]" + "\n")
            file.write("Old view sem diff avg: "+ str(np.mean(sem_diff_old))+ "\n")
            file.write("[" + ' '.join(str(v) for v in sem_diff) + "]" + "\n")
            file.write("Novel view sem diff avg: "+ str(np.mean(sem_diff))+ "\n\n")
            file.write("--------------------------------- \n\n")

        self.model.train()

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        if self.opt.latent and self.opt.latent_space == 'original':
            H = H/8
            W = W/8
        elif self.opt.low_res_img:
            H = H/8
            W = W/8

        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, preds_sem, preds_depth_ori = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            if self.vae is None:
                preds_sem = F.interpolate(preds_sem.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
                preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        preds_depth_ori = preds_depth_ori[0].detach().cpu().numpy()
        if self.vae is not None:
            outputs = {
                'image': pred,
                'depth': pred_depth,
                'depth_ori': preds_depth_ori
            }
            return outputs
    
        else:
            outputs = {
                'image': pred,
                'depth': pred_depth,
                'depth_ori': preds_depth_ori
            }

            return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if os.environ.get('ONLYCOORD', False):
            self.log("[INFO] In the mode of ONLYCOORD, not loading optimizer for regularization test.")
        else:
            if self.optimizer and 'optimizer' in checkpoint_dict:
                try:
                    self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                    self.log("[INFO] loaded optimizer.")
                except:
                    self.log("[WARN] Failed to load optimizer.")
            
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")