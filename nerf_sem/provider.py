import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from torchvision import transforms
import trimesh
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from .utils import get_rays

from scipy import interpolate
from skimage.transform import resize_local_mean
from PIL import Image
# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, val_sample_step=20, vae=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.

        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.vae = vae

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.val_sample_step = val_sample_step
        # load nerf-compatible format data.
        with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
            transform = json.load(f)

        # replica apartment 2 dining room
        test_index = [404, 428]
        verify_index = [50,100,150,200,230,300,350,400]
        test_index = [70, 72]
        verify_index = [10,25,50,60]

        self.test_len = len(test_index)

        self.target_labels_ins = set() # only for classification sem model
        self.target_labels_sem = set() # only for classification sem model

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        if self.opt.low_res_img:
            self.H_ray = int(self.H/8)
            self.W_ray = int(self.W/8)
        else:
            self.H_ray = self.H 
            self.W_ray = self.W
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # TODO: hard to get correct scale_factor
        self.scale_factor = transform['scale_factor'] * self.scale

        # for colmap, manually interpolate a test set.
        if type == 'test':
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
        else:
            self.poses = []
            self.images = []
            self.depths = []

            self.extra_inputs = []

            for k, f in enumerate(tqdm.tqdm(frames, desc=f'Loading {type} data')):
                img_idx = f['file_path'].split('/')[-1].split('.')[0]
                # NOTE: Load RGB
                # there are non-exist paths in fox...
                f_path = os.path.join(self.root_path, f['file_path']) # rgb
                if not os.path.exists(f_path):
                    raise FileNotFoundError(f"Please check transforms.json file_path = {f_path}")
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                image = image.astype(np.float32) / 255 # [H, W, 3/4]
                image = image[:,:,:3]

                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # NOTE: Load depth
                f_path_depth = os.path.join(self.root_path, 'depth', f"{img_idx}.npy")
                depth_data = np.load(f_path_depth)
                depth_data[depth_data==0] = 20000 # TODO give large value for inf depth 
                
                if os.environ.get('DEBUG', False):
                    def depth_observation(depth_obs):
                        depth_img = Image.fromarray((depth_obs / np.amax(depth_obs) * 255).astype(np.uint8), mode="L")
                        depth_img.show()
                    depth_observation(depth_data)

                if depth_data.shape[0] != self.H or depth_data.shape[1] != self.W:
                    depth_data = resize_local_mean(depth_data, (self.H, self.W), preserve_range=True)
                
                depth_data = depth_data / 1000 * self.scale_factor
                
                f = transform['fl_x']
                if self.opt.radial_depth:
                    for i in range(self.W):
                        for j in range(self.H):
                            depth_data[j, i] = np.sqrt(f**2 + (i-transform['cx'])**2 + (j-transform['cy'])**2) * depth_data[j,i]/f
                else:
                    xs, ys = np.meshgrid(np.arange(self.W), np.arange(self.H))
                    self.depth_radial2plane = f/np.sqrt(f**2 + (xs - transform['cx']) ** 2 + (ys - transform['cy']) ** 2)

                # NOTE: Load Extra
                extra_in = {}
                if self.opt.load_sem:
                    # semantic segmentation
                    f_path_sem = os.path.join(self.root_path, 'sem', f"{img_idx}-label.npy")
                    sem_data = np.load(f_path_sem)
                    self.target_labels_sem = self.target_labels_sem.union(set(np.unique(sem_data)))
                    sem_data[sem_data==-100] = 0
                    if sem_data.shape[0] != self.H or sem_data.shape[1] != self.W:
                        sem_data = cv2.resize(sem_data, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                    extra_in['sem_map'] = sem_data
                    
                    # instance segmentation
                    f_path_sem = os.path.join(self.root_path, 'sem', f"{img_idx}-instance.npy")
                    sem_data2 = np.load(f_path_sem)
                    self.target_labels_ins = self.target_labels_ins.union(set(np.unique(sem_data2)))
                    sem_data2[sem_data2==-100] = 0
                    if sem_data2.shape[0] != self.H or sem_data2.shape[1] != self.W:
                        sem_data2 = cv2.resize(sem_data2, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                    extra_in['ins_map'] = sem_data2
                else:
                    pass
                
                if self.opt.use_normal:
                    f_path_norm = os.path.join(self.root_path, 'normal', f"{img_idx}.npy")
                    normal_data = np.load(f_path_norm, allow_pickle=True)
                    normal = normal_data.item().get('normal')
                    msk = ~normal_data.item().get('msk')
                    if normal.shape[0] != self.H or normal.shape[1] != self.W:
                        normal = cv2.resize(normal, (self.W, self.H), interpolation=cv2.INTER_AREA)
                        msk = cv2.resize(msk, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    extra_in['normal_map'] = normal
                    extra_in['normal_msk'] = msk

                self.poses.append(pose)
                self.images.append(image)
                self.depths.append(depth_data)
                self.extra_inputs.append(extra_in)
        
        # Create data splits
        self.poses_train = []
        self.images_train = []
        self.depths_train = []
        self.extra_inputs_train = []

        self.poses_test = []
        self.images_test = []
        self.depths_test = []
        self.extra_inputs_test = []

        for k in range(len(self.poses)):
            if k in test_index:
                self.poses_test.append(self.poses[k])
                self.images_test.append(self.images[k])
                self.extra_inputs_test.append(self.extra_inputs[k])
                self.depths_test.append(self.depths[k])
            else:
                self.poses_train.append(self.poses[k])
                self.images_train.append(self.images[k])
                self.extra_inputs_train.append(self.extra_inputs[k])
                self.depths_train.append(self.depths[k])

        # NOTE: copy a set of frames as verification of training quality
        self.poses_verify = [p for i, p in enumerate(self.poses_train) if i in verify_index ]
        self.images_verify = [im for i, im in enumerate(self.images_train) if i in verify_index ]
        self.depths_datas_verify = [im for i, im in enumerate(self.depths_train) if i in verify_index ]
        self.extra_inputs_verify = [im for i, im in enumerate(self.extra_inputs_train) if i in verify_index ]
        self.poses_verify += self.poses_test
        self.images_verify += self.images_test
        self.depths_datas_verify += self.depths_test
        self.extra_inputs_verify += self.extra_inputs_test

        self.poses_train = torch.from_numpy(np.stack(self.poses_train, axis=0)) # [N, 4, 4]
        self.images_train = torch.from_numpy(np.stack(self.images_train, axis=0)) # [N, H, W, C]
        self.depths_train = torch.from_numpy(np.stack(self.depths_train, axis=0))
        self.depth_radial2plane = torch.from_numpy(self.depth_radial2plane)

        # NOTE:
        self.extra_in_train = {}
        for k in self.extra_inputs_train[0].keys():
            self.extra_in_train[k] = torch.from_numpy(np.stack([info[k] for info in self.extra_inputs_train], axis=0))
        
        self.radius = self.poses_train[:, :3, 3].norm(dim=-1).mean(0).item()

        self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())
        if self.preload:
            self.poses_train = self.poses_train.to(self.device)
            if self.images_train is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images_train = self.images_train.to(dtype).to(self.device)
                self.depths_train = self.depths_train.to(dtype).to(self.device)
                self.depth_radial2plane = self.depth_radial2plane.to(dtype).to(self.device)

                for k in self.extra_in_train.keys():
                    self.extra_in_train[k] = self.extra_in_train[k].to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
 
    @property
    def num_labels(self):
        if len(self.target_labels) == 0:
            return 3
        return max(self.target_labels) + 1

    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H_ray * self.W_ray / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H_ray / s), int(self.W_ray / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses_train[index].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.intrinsics, self.H_ray, self.W_ray, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H_ray,
            'W': self.W_ray,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'pixel_space_inds': rays['pixel_space_inds']
        }

        images = self.images_train[index].to(self.device) # [B, H, W, 3/4]
        if self.opt.low_res_img:
            images = images[:, ::8, ::8, :].contiguous()

        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        results['images'] = images
        
        depths = self.depths_train[index].to(self.device) # [B, H, W, none/3/4]
        depth_radial2plane = self.depth_radial2plane

        if self.training:
            C = depths.shape[-1]
            depths = torch.gather(depths.view(B, -1), 1, rays['inds'])
            depth_radial2plane = torch.gather(depth_radial2plane.view(B, -1), 1, rays['inds'])
        results['images_depth'] = depths 
        results['depth_radial2plane'] = depth_radial2plane 

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
        
        if self.extra_in_train is not None:
            for k in self.extra_in_train:
                datum = self.extra_in_train[k][index].to(self.device) # [B, H, W, none/3/4]

                if self.training:
                    C = datum.shape[-1]
                    if len(datum.shape) == 3:
                        datum = torch.gather(datum.view(B, -1), 1, rays['inds']) # [B, N]
                    else:
                        datum = torch.gather(datum.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                results[k] = datum 
        
        return results

    def dataloader(self):
        # size = len(self.poses) # leak test data
        size = len(self.poses_train)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images_train is not None
        return loader