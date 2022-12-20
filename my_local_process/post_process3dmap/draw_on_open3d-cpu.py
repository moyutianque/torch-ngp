import open3d as o3d
import numpy as np
import argparse
from scipy.ndimage import maximum_filter, median_filter
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm
from utils import merge_volume, split_volume, zero_pads, zero_unpads, entropy_filter, mode_filter
import os
import cc3d # connected components in 3D

def process_density(dm_np, t1 = 5.):
    """ return which volumns are possibly occupied """
    return dm_np > t1

def process_sem(sm_np, downsample_scale, msk2):
    """ generate semantic map volumn """
    sm_np[~msk2] = 0
    sm_np = mode_filter(sm_np.astype(int), kernel_size=downsample_scale, downsample_scale=downsample_scale)
    # sm_np = maximum_filter(sm_np, size=downsample_scale, mode='constant')
    # sm_np = sm_np[::downsample_scale, ::downsample_scale, ::downsample_scale]
    return sm_np

def filter_dust(sm_np):
    sem_labels = np.unique(sm_np)
    print(sem_labels)
    msk = None
    for label in sem_labels:
        if label == 0:
            continue
        
        binary_msk = (sm_np == label)
        # using connected components for filtering dust
        labels_out, N = cc3d.largest_k(
                binary_msk, k=1, 
                connectivity=6, delta=0,
                return_N=True,
            )
        
        if N == 1:
            if msk is not None:
                msk = msk | (labels_out == 1)
            else:
                msk = (labels_out == 1)
    return sm_np * msk

def draw_map(m):
    sem_labels = np.unique(m)
    mapper = {l:i for i, l in enumerate(sorted(sem_labels))}
    print(mapper)
    m = np.vectorize(mapper.get)(m)

    if os.environ.get('DEBUG', False):
        m[0:20, 20:40, 0:20] = 1 # debug coordinate

    pcd = o3d.geometry.PointCloud()
    xx,yy,zz =np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), np.arange(m.shape[2]), indexing='ij')
    
    pcd_np = np.vstack((xx.ravel(), yy.ravel(), zz.ravel(), m.ravel())).T

    pcd_msk = pcd_np[:,3] > 0
    pcd_np = pcd_np[pcd_msk]
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])

    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, len(mapper)))

    while True:
        np.random.shuffle(color_palette)
        facecolors = color_palette[m]
        facecolors = np.reshape(facecolors, (-1, 4))[pcd_msk]
        pcd.colors = o3d.utility.Vector3dVector(facecolors[:,:3])

        # pcd, ind = pcd.remove_radius_outlier(nb_points=70, radius=5)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        o3d.visualization.draw_geometries([voxel_grid], width=700, height=700)

        # o3d.io.write_point_cloud("./sem_map.ply", pcd)    
        in_key = input()
        if in_key.strip() == 'q':
            break
    
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
    dm = np.log(dm)
    sm = filter_map(sm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)

    if os.environ.get('DEBUG', False):
        dm_coord = [0, 80, 0]
        step = 5

        test_chunck = dm[
            dm_coord[0]: dm_coord[0]+step, 
            dm_coord[1]: dm_coord[1]+step, 
            dm_coord[2]: dm_coord[2]+step
        ]
        test_chunck_sem = sm[
            dm_coord[0]: dm_coord[0]+step, 
            dm_coord[1]: dm_coord[1]+step, 
            dm_coord[2]: dm_coord[2]+step
        ]
        test_chunck = test_chunck.astype(int)
        import ipdb;ipdb.set_trace() # breakpoint 129
        print(np.unique(dm.astype(int)))
        print()

    # x_range = np.log(np.amax(dm))
    # y = np.zeros((int(x_range)+1))
    # for item in tqdm(np.random.choice(dm.flatten(), size=int(len(dm.flatten())/10))):
    #     if item<1:
    #         item = 1
    #     y[int(np.log(item))] += 1
    # plt.bar(np.arange(x_range), y)
    # plt.show()
    # import ipdb;ipdb.set_trace() # breakpoint 102

    # mask = dm > opt.d_thresh
    # sm[~mask] = 0
    with torch.no_grad():
        d_msk = process_density(dm, t1=opt.t1)
        sm_out = process_sem(sm, downsample_scale=opt.downsample_scale, msk2=d_msk)
        # sm_out = filter_dust(sm_out)
    draw_map(sm_out)
