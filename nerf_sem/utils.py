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
    results['pixel_space_inds'] =None
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
            results['pixel_space_inds'] = inds

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

def extract_fields_sem(bound_min, bound_max, resolution, query_func, S=128, return_pts=False):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u_density = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    if return_pts:
        u_pts = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)

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
                 criterion_ce=None,
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
                 vae=None,
                 extra_configs=None
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
        self.extra_configs = extra_configs

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        
        self.criterion = criterion
        self.criterion_ce = criterion_ce

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)
        
        if os.environ.get('USE_UNet', False):
            from .UNet import PatchFeaUNet
            self.conv = PatchFeaUNet(32, 3).cuda()
        elif os.environ.get('USE_Latent_DOWN8x', False):
            from .UNet.unet_parts import down
            self.conv = nn.ModuleList([
                    down(extra_configs[0].dim_out, 16),
                    down(16, 8),
                    down(8, 4)
                ]
            ).cuda()
    
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
            elif self.use_checkpoint == "base":
                self.log("[INFO] Loading base checkpoint ...")
                self.load_checkpoint(os.path.join(self.ckpt_path, 'base.pth'))
                if os.environ.get('INCLUDE_LATENT_LOSS', False):
                    self.optimizer.add_param_group({"params": self.model.extra_nets.parameters(), 'lr': self.opt.lr_latent})
                    self.optimizer.add_param_group({"params": self.conv.parameters(), "lr": self.opt.lr_latent})
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...
        
        normal_net_dim_in = 1
        if self.opt.sem_label:
            print('add parameter groups sem_label_emb successfully')
            self.optimizer.add_param_group({"params": self.model.sem_label_emb.parameters(), "lr": self.opt.lr})
            normal_net_dim_in += 16
        if self.opt.sem_ins:
            print('add parameter groups sem_ins_emb successfully')
            self.optimizer.add_param_group({"params": self.model.sem_ins_emb.parameters(), "lr": self.opt.lr})
            normal_net_dim_in += 16
        
        if self.opt.use_normal:
            from .UNet import PatchFeaUNet
            # self.norm_net = PatchFeaUNet(normal_net_dim_in, 3, act='tanh').cuda()
            self.norm_net = PatchFeaUNet(normal_net_dim_in, 3, act='none').cuda()
            self.optimizer.add_param_group({"params": self.norm_net.parameters(), "lr": self.opt.lr})
            print('add parameter groups norm_net successfully')
        
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
        gt_rgb = data['images'] # [B, N, 3/4]

        bg_color=1
        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, global_step=self.global_step, extra_configs=self.extra_configs,**vars(self.opt))

        pred_rgb = outputs['image'] 

        # NOTE by zehao, default this is not called
        # # patch-based rendering
        if self.opt.patch_size > 1: 
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            loss = self.criterion(pred_rgb, gt_rgb).mean((1,2,3))
        else:
            # MSE loss
            loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss_rgb = loss.mean()

        loss_depth = torch.tensor(0.).cuda()
        if self.opt.use_depth:
            pred_depth = outputs['depth'] * data['depth_radial2plane']
            pred_depth = pred_depth.view(-1, self.opt.patch_size, self.opt.patch_size, 1).permute(0, 3, 1, 2).contiguous()
            gt_depth = data['images_depth'].view(-1, self.opt.patch_size, self.opt.patch_size, 1).permute(0, 3, 1, 2).contiguous()
            
            loss_depth = torch.abs(torch.log(gt_depth)-torch.log(pred_depth))
            filtered_idx = (~torch.isinf(loss_depth)) & (~torch.isnan(loss_depth))
            loss_depth = loss_depth[filtered_idx].mean()

            # NOTE: simple version
            # loss_depth = self.criterion(pred_depth, gt_depth).mean()

        loss_normal = torch.tensor(0.).cuda()
        if self.opt.use_normal:
            pred_depth = outputs['depth'] * data['depth_radial2plane']
            pred_depth = pred_depth.view(-1, self.opt.patch_size, self.opt.patch_size, 1).permute(0, 3, 1, 2).contiguous()
            gt_normal = data['normal_map']
            gt_normal_msk = data['normal_msk']
            gt_normal = gt_normal.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            gt_normal_msk = gt_normal_msk.view(-1, self.opt.patch_size, self.opt.patch_size, 1).permute(0, 3, 1, 2).contiguous()
            gt_normal_msk = gt_normal_msk.repeat(1,3,1,1)
            
            if self.opt.sem_label:
                gt_sem_label = data['sem_map']
                gt_sem_label = gt_sem_label.view(-1, self.opt.patch_size, self.opt.patch_size).contiguous()
                gt_sem_label = self.model.sem_label_emb(gt_sem_label).permute(0,3,1,2)
                pred_depth = torch.cat([pred_depth, gt_sem_label], dim=1)
            
            indices = {i for i in range(len(gt_normal))}
            if self.opt.sem_ins:
                gt_sem_ins = data['ins_map']
                gt_sem_ins = gt_sem_ins.view(-1, self.opt.patch_size, self.opt.patch_size).contiguous()
                
                # NOTE: filter out patch who only has one label
                for i, sem_ins in enumerate(gt_sem_ins):
                    if len(torch.unique(sem_ins)) <= 1:
                        indices.remove(i)

                gt_sem_ins = self.model.sem_ins_emb(gt_sem_ins).permute(0,3,1,2)
                pred_depth = torch.cat([pred_depth, gt_sem_ins], dim=1)

            indices = np.array(list(indices))   

            if len(indices)> 0: 
                pred_depth = pred_depth[indices, :, :, :]
                gt_normal_msk = gt_normal_msk[indices, :, :, :]
                gt_normal = gt_normal[indices, :, :, :]

                pred_norm = self.norm_net(pred_depth)
                if torch.any(gt_normal_msk):
                    loss_normal = torch.tensor(0.).cuda()
                    cnt=0
                    for i in range(pred_norm.shape[0]):
                        if torch.any(gt_normal_msk[i]):
                            cnt += 1
                            loss_normal += self.criterion(pred_norm[i][:,1:-1,1:-1][gt_normal_msk[i][:,1:-1,1:-1]], gt_normal[i][:,1:-1,1:-1][gt_normal_msk[i][:,1:-1,1:-1]].to(gt_rgb.dtype)).mean()
                    loss_normal = loss_normal/cnt

        # TODO: make multiple extra outs possible
        extra_losses = []
        extra_outs = outputs['extra_outs']    
        for extra_out in extra_outs:
            extra_out = extra_out.view(-1, self.opt.patch_size, self.opt.patch_size, extra_out.shape[-1]).permute(0, 3, 1, 2).contiguous() 

            # remove UNet
            if os.environ.get('USE_UNet', False):
                pred_extra = self.conv(extra_out)
                loss = self.criterion(pred_extra, gt_rgb).mean((1,2,3))
            else:
                # feat space to pre-trained latent space
                for layer in self.conv:
                    extra_out = layer(extra_out)
                pred_extra = extra_out

                if os.environ.get('VIS_PATCH', False):
                    import matplotlib.pyplot as plt
                    for gt_rgb_item in gt_rgb:
                        plt.imshow(gt_rgb_item.permute(1,2,0).cpu().numpy().astype(float))
                        plt.show()
                target_latents = self.vae.encode(gt_rgb).latent_dist.mode()
                
                loss = self.criterion(pred_extra, target_latents).mean((1,2,3))
            extra_losses.append(loss.mean())
        loss = {
            'loss_rgb': loss_rgb,
            'loss_depth': loss_depth,
            'loss_normal': loss_normal,
            "extra_losses": extra_losses,
        }

        return pred_rgb, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, extra_in=None):  
        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, global_step=self.global_step, extra_configs=self.extra_configs, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth_normalized = outputs['depth_normalized'].reshape(-1, H, W)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        extras = []
        if self.opt.use_normal:
            assert extra_in is not None
            plane_depth = pred_depth * data['depth_radial2plane']
            plane_depth = plane_depth[None, ...]
            gt_normal = extra_in['normal_map']
            gt_normal_msk = extra_in['normal_msk']
            if self.opt.sem_label:
                gt_sem_label = extra_in['sem_map']
                gt_sem_label = torch.from_numpy(gt_sem_label)[None, ...].cuda()
                gt_sem_label = self.model.sem_label_emb(gt_sem_label).permute(0,3,1,2)
                plane_depth = torch.cat([plane_depth, gt_sem_label], dim=1)
            
            if self.opt.sem_ins:
                gt_sem_ins = extra_in['ins_map']
                gt_sem_ins = torch.from_numpy(gt_sem_ins)[None, ...].cuda()
                gt_sem_ins = self.model.sem_ins_emb(gt_sem_ins).permute(0,3,1,2)
                plane_depth = torch.cat([plane_depth, gt_sem_ins], dim=1)
            
            pred_norm = self.norm_net(plane_depth)
            pred_norm += 1
            pred_norm /= 2
            pred_norm[:, :, ~gt_normal_msk] = 0
            extras.append(pred_norm)
        else:
            for extra_out in outputs['extra_outs'] :
                extra_out = extra_out.view(-1, H, W, extra_out.shape[-1]).permute(0, 3, 1, 2).contiguous() 
                if os.environ.get('USE_UNet', False):
                    decode_rgb = self.conv(extra_out)
                else:
                    for layer in self.conv:
                        extra_out = layer(extra_out)
                        decode_rgb = self.vae.decode(extra_out)
                extras.append(decode_rgb)

        return pred_rgb, pred_depth_normalized, pred_depth, extras


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

    def get_3dmap(self, resolution, return_pts=False):
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    density_outputs = self.model.density(pts.to(self.device))
                    sem_out = self.model.sem(pts.to(self.device), **density_outputs)
            return density_outputs['sigma'], sem_out
        
        return extract_fields_sem(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, query_func=query_func, return_pts=return_pts)

    def save_3dmap(self, resolution=256):
        save_path = os.path.join(self.workspace, f'{self.name}_3dmap.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        density_out, sem_out = self.get_3dmap(resolution, return_pts=False)

        with h5py.File(save_path, "w") as f:
            f.create_dataset("density", data=density_out)
            f.create_dataset("sem", data=sem_out)

    ### ------------------------------

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
        
    def train(self, train_loader, step=16, val_data=None):
        self.model.train()

        total_loss_rgb = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_loss_normal = torch.tensor([0], dtype=torch.float32, device=self.device)
        total_loss_extra = torch.tensor([0], dtype=torch.float32, device=self.device)
        
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

            tot_loss = loss_dict['loss_rgb'].clone()

            if self.global_step > self.opt.warmup_iter:
                if self.opt.use_normal:
                    tot_loss += loss_dict['loss_normal']
                if self.opt.use_depth:
                    tot_loss += loss_dict['loss_depth']

                if os.environ.get('INCLUDE_LATENT_LOSS', False):
                    for extra_loss in loss_dict['extra_losses']:
                        tot_loss += extra_loss

            self.scaler.scale(tot_loss).backward()
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
 
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            if os.environ.get('INCLUDE_LATENT_LOSS', False):
                total_loss_extra += loss_dict['extra_losses'][0].detach()
            if 'loss_rgb' in loss_dict:
                total_loss_rgb += loss_dict['loss_rgb'].detach()
            
            if 'loss_normal' in loss_dict:
                total_loss_normal += loss_dict['loss_normal'].detach()

            if self.global_step != 0 and (self.global_step % self.opt.save_iter == 0):
                self.get_view_sythesis(val_data, train_loader._data.intrinsics, train_loader._data.test_len, self.global_step, train_loader._data.depth_radial2plane)
                self.model.training = True
                self.save_checkpoint(full=True, best=False, remove_old=False)
                self.epoch += 1

        if self.ema is not None:
            self.ema.update()

        rgb_loss = total_loss_rgb.item() / step
        normal_loss = total_loss_normal.item() / step
        extra_loss_record = total_loss_extra.item() / step

        outputs = {
            'l_rgb': rgb_loss,
            'l_normal': normal_loss,
            'l_extra': extra_loss_record,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs    

    def get_view_sythesis(self, val_data, intrinsics, test_len, iters, depth_radial2plane=None):
        val_results = []
        save_path = os.path.join(self.workspace, 'vis_eval')
        os.makedirs(save_path, exist_ok=True)
        psnr_list = []
        psnr_list_oldview = []
        for j, item in enumerate(zip(*val_data)):
            (p, im, depth, extra_im) = item
            test_output = self.test_gui(
                    p, intrinsics, 
                    im.shape[1], im.shape[0], bg_color=torch.ones(3).float(), 
                    depth_radial2plane=depth_radial2plane,
                    extra_in = extra_im
                )

            img_rgb = (im * 255).astype(np.uint8)
            pred = (test_output['image'] * 255).astype(np.uint8)
            pred_depth = (test_output['depth'] * 255).astype(np.uint8)
            depth_ori = test_output['depth_ori']

            if self.opt.use_normal:
                gt_normal = ((extra_im['normal_map'] + 1)/2 * 255).astype(np.uint8)

            decoded_rgb = None
            if test_output['decoded_rgb'] is not None:
                decoded_rgb = (test_output['decoded_rgb'] * 255).astype(np.uint8)
            
            prefix = ''
            if len(val_data[0])-j <= test_len:
                prefix = 'newview'
                # psnr_list.append(np_PSNR(img_rgb, pred))
            else:
                # psnr_list_oldview.append(np_PSNR(img_rgb, pred))
                pass

            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}gt.png'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_rgb_{prefix}.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            if decoded_rgb is not None:
                cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_decoded_rgb_{prefix}.png'), cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_depth_{prefix}.png'), pred_depth)
            if self.opt.use_normal:
                cv2.imwrite(os.path.join(save_path, f'{iters}_{j}_normal_{prefix}.png'), cv2.cvtColor(gt_normal, cv2.COLOR_RGB2BGR))
            val_results.append(test_output)

        self.model.train()

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, depth_radial2plane=None, extra_in=None):
        
        # render resolution (may need downscale to for better frame rate)
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
        if depth_radial2plane is not None:
            data.update({"depth_radial2plane": depth_radial2plane})
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, preds_depth_ori, extras = self.test_step(data, extra_in=extra_in, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        preds_depth_ori = preds_depth_ori[0].detach().cpu().numpy()

        decoded_rgb = None
        if len(extras) != 0:
            decoded_rgb = extras[0][0].detach().permute(1,2,0).cpu().numpy()
        
        outputs = {
            'image': pred,
            'depth': pred_depth,
            'depth_ori': preds_depth_ori,
            'decoded_rgb': decoded_rgb
        }
        return outputs
    
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