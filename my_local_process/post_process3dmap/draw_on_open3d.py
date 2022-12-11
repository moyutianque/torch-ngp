import open3d as o3d
import numpy as np
import argparse
from scipy.ndimage import maximum_filter, median_filter
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm
from utils import merge_volume, split_volume, zero_pads, zero_unpads

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
    return msk

def process_sem(sm_np, downsample_scale, msk2, t3=0.5):
    """ generate semantic map volumn """
    sm = torch.from_numpy(sm_np).cuda().long()[None, ...]
    sm2 = zero_pads(sm, voxel_size=downsample_scale)
    del sm
    torch.cuda.empty_cache()

    sm_split, sm_size = split_volume(sm2, voxel_size=downsample_scale)
    sm_split = sm_split.view(-1, downsample_scale*downsample_scale*downsample_scale)

    sm_mode_out, _ = torch.mode(sm_split, dim=1)
    sm_mode = sm_mode_out.unsqueeze(1).repeat(1, sm_split.shape[1])

    mode_tr = torch.sum(sm_split == sm_mode, dim=1)/sm_split.shape[1]
    msk3 = (mode_tr > t3) & (sm_mode_out!=0)

    print(torch.sum(msk3 ))
    print(torch.sum(msk2 & msk3))

    sm_mode_out[~(msk2 & msk3)] = 0
    sm_mode_out = sm_mode_out.view(*sm_size)
    return sm_mode_out

def draw_map(m):
    m = m.astype(int)
    sem_labels = np.unique(m)

    mapper = {l:i for i, l in enumerate(sorted(sem_labels))}

    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, len(mapper)))
    np.random.shuffle(color_palette)

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
    
    parser.add_argument('--t1', type=float, default=5)
    parser.add_argument('--t2', type=float, default=0.5)
    parser.add_argument('--t3', type=float, default=0.5)
    
    parser.add_argument('--downsample_scale', type=int, default=4)
    parser.add_argument('--d_thresh', type=float, default=1)
    parser.add_argument('--map_path', type=str, default='../../outputs/replica_apart2_dinning_room/ngp_3dmap-512.npy')
    opt = parser.parse_args()

    dm, sm = load_map(opt.map_path)
    dm = filter_map(dm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)
    sm = filter_map(sm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)

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
        d_msk = process_density(dm, downsample_scale=opt.downsample_scale, t1=opt.t1, t2=opt.t2)
        sm_out = process_sem(sm, downsample_scale=opt.downsample_scale, msk2=d_msk, t3=opt.t3)
    sm_out = sm_out.cpu().numpy()
    draw_map(sm_out)
