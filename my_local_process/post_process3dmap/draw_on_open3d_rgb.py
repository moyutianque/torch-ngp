import open3d as o3d
import numpy as np
import argparse
from scipy.ndimage import maximum_filter, median_filter
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm
from utils import merge_volume, split_volume, zero_pads, zero_unpads, zero_pads_np

def process_density(dm_np, downsample_scale, t1 = 5., t2 = 0.5):
    """ return which volumns are possibly occupied """
    dm = torch.from_numpy(dm_np).cuda()[None, ...]
    dm2 = zero_pads(dm,  voxel_size=downsample_scale)
    del dm
    torch.cuda.empty_cache()

    dm_split, dm_size = split_volume(dm2, voxel_size=downsample_scale)
    del dm2
    torch.cuda.empty_cache()

    dm_split = dm_split.view(-1, downsample_scale*downsample_scale*downsample_scale)
    msk = torch.sum(dm_split > t1, dim=1)/dm_split.shape[1] > t2

    del dm_split
    torch.cuda.empty_cache()
    
    # msk_ratio = torch.sum(msk, dim=1)/msk.shape[1]
    # msk2 = msk_ratio > t2
    return msk.cpu().numpy(), dm_size

def process_sem(sm_np, dm_size, downsample_scale, msk2, t3=0.5):
    """ generate semantic map volumn """
    sm_np = zero_pads_np(sm_np, voxel_size=downsample_scale)
    sm_np = sm_np[::downsample_scale, ::downsample_scale, ::downsample_scale]

    assert dm_size[0] == sm_np.shape[0]
    assert dm_size[1] == sm_np.shape[1]
    assert dm_size[2] == sm_np.shape[2]

    msk = np.abs(np.sum(np.resize(sm_np, (int(np.prod(sm_np.shape[:3])), 3)) - [0,0,0], axis=1)) > 1e-5

    voxel_exist_msk = msk2 & msk

    return np.reshape(sm_np, (dm_size[0], dm_size[1], dm_size[2], 3)), voxel_exist_msk

def draw_map(m, voxel_exist_msk):
    facecolors = m

    pcd = o3d.geometry.PointCloud()
    xx,yy,zz =np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), np.arange(m.shape[2]), indexing='ij')
    
    pcd_np = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    pcd_msk = voxel_exist_msk
    pcd_np = pcd_np[pcd_msk]
    facecolors = (np.reshape(facecolors, (-1, 3))[pcd_msk])

    pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(facecolors[:,:3])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])

    # o3d.io.write_point_cloud("./sem_map.ply", pcd)
    o3d.io.write_voxel_grid("./sem_map.ply", voxel_grid)

def load_map(map_path):
    with h5py.File(map_path, "r") as f:
        density_map = f['density'][()]
        sem_map = f['sem'][()]
    # map_np = np.load(map_path, allow_pickle=True)[()]
    # return map_np['density'], map_np['sem']
    return density_map, sem_map

def filter_map(m,bx,by,bz,tx,ty,tz):
    x_max, y_max, z_max = m.shape
    return m[int(x_max* bx):int(x_max * tx), int(y_max * by):int(y_max * ty), int(z_max * bz):int(z_max * tz)]

def filter_map_sem(m,bx,by,bz,tx,ty,tz):
    x_max, y_max, z_max, _ = m.shape
    return m[int(x_max* bx):int(x_max * tx), int(y_max * by):int(y_max * ty), int(z_max * bz):int(z_max * tz)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bx', type=float, default=0)
    parser.add_argument('--by', type=float, default=0)
    parser.add_argument('--bz', type=float, default=0)
    parser.add_argument('--tx', type=float, default=1)
    parser.add_argument('--ty', type=float, default=1)
    parser.add_argument('--tz', type=float, default=1)
    
    parser.add_argument('--t1', type=float, default=5)
    parser.add_argument('--t2', type=float, default=0.5)
    parser.add_argument('--t3', type=float, default=0.5)
    
    parser.add_argument('--downsample_scale', type=int, default=4)
    parser.add_argument('--d_thresh', type=float, default=1)
    parser.add_argument('--map_path', type=str, default='../../outputs/replica_apart2_dinning_room/ngp_3dmap-512.npy')
    opt = parser.parse_args()

    dm, sm = load_map(opt.map_path)
    dm = filter_map(dm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)
    sm = filter_map_sem(sm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)
    
    with torch.no_grad():
        d_msk, dm_size = process_density(dm, downsample_scale=opt.downsample_scale, t1=opt.t1, t2=opt.t2)
        del dm
        sm_out, voxel_exist_msk = process_sem(sm, dm_size, downsample_scale=opt.downsample_scale, msk2=d_msk, t3=opt.t3)
        del sm

    draw_map(sm_out, voxel_exist_msk)
