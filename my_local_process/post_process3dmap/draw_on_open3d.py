import open3d as o3d
import numpy as np
import argparse
from scipy.ndimage import maximum_filter, median_filter, generic_filter
import matplotlib.pyplot as plt
import h5py
import torch

# from utils import merge_volume, split_volume

from scipy import stats
def mode_filter_func(p):
    """ cope with generic_filter, but only on cpu """
    # p has the length of prod(filtered region), if 3d, then = size*size*size
    mode = stats.mode(p)
    return mode.mode[0]

def draw_map(m, mask, downsample_scale):

    # m = torch.from_numpy(m)

    m = m.astype(int)
    sem_labels = np.unique(m[mask])

    # m = maximum_filter(m, size=downsample_scale, mode='constant')
    m = generic_filter(m, mode_filter_func, size=downsample_scale, mode='constant') # very slow, not parallel
    m = m[::downsample_scale, ::downsample_scale, ::downsample_scale]

    mapper = {l:i for i, l in enumerate(sorted(sem_labels))}

    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, len(mapper)))
    m = np.vectorize(mapper.get)(m)
    facecolors = color_palette[m]

    pcd = o3d.geometry.PointCloud()
    xx,yy,zz =np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), np.arange(m.shape[2]), indexing='ij')
    
    pcd_np = np.vstack((xx.ravel(), yy.ravel(), zz.ravel(), m.ravel())).T


    pcd_msk = pcd_np[:,3]!=0
    pcd_np = pcd_np[pcd_msk]
    facecolors = np.reshape(facecolors, (-1, 4))[pcd_msk]

    pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(facecolors[:,:3])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=2)
    o3d.visualization.draw_geometries([voxel_grid])

    # o3d.io.write_point_cloud("./sem_map.ply", pcd)
    # o3d.io.write_voxel_grid("./sem_map.ply", voxel_grid)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bx', type=float, default=0)
    parser.add_argument('--by', type=float, default=0)
    parser.add_argument('--bz', type=float, default=0)
    parser.add_argument('--tx', type=float, default=1)
    parser.add_argument('--ty', type=float, default=1)
    parser.add_argument('--tz', type=float, default=1)
    parser.add_argument('--d_thresh', type=float, default=1)
    parser.add_argument('--map_path', type=str, default='../../outputs/replica_apart2_dinning_room/ngp_3dmap-512.npy')
    opt = parser.parse_args()

    dm, sm = load_map(opt.map_path)
    dm = filter_map(dm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)
    sm = filter_map(sm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)

    mask = dm > opt.d_thresh
    sm[~mask] = 0
    draw_map(sm, mask, downsample_scale=4)
