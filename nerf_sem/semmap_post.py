import open3d as o3d
import numpy as np
import torch
import cc3d # connected components in 3D
import numpy as np
import skimage
import multiprocessing
import matplotlib.pyplot as plt
import os 

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def mode_func(labels):
    # pure mode operation
    vals, cnts = np.unique(labels, return_counts=True)
    if vals[0] == 0:
        if len(vals) == 1:
            return 0
        cnts = cnts[1:]
        vals = vals[1:]
    return vals[0]

def mode_filter(np_arr, kernel_size, downsample_scale=4):
    pad_size = int(kernel_size//2)
    np_arr_chunks = skimage.util.view_as_windows(np.pad(np_arr, (pad_size, pad_size), mode='constant'), window_shape=(kernel_size, kernel_size, kernel_size))
    np_arr_chunks = np_arr_chunks[::downsample_scale,::downsample_scale,::downsample_scale]
    out_shape = np_arr_chunks.shape[:3]
    np_arr_chunks = np.reshape(np_arr_chunks, (int(np.prod(out_shape)), -1))
    # out = parallel_apply_along_axis(mode_cus, axis=1, arr=np_arr_chunks, threshold=1/kernel_size)
    out = parallel_apply_along_axis(mode_func, axis=1, arr=np_arr_chunks)
    return np.reshape(out, out_shape)

def process_density(dm_np, t1 = 5.):
    """ return which volumns are possibly occupied """
    return dm_np > t1

def process_sem(sm_np, downsample_scale, msk2):
    """ generate semantic map volumn """
    sm_np[~msk2] = 0

    # TODO include post downsample
    # sm_np = mode_filter(sm_np.astype(int), kernel_size=downsample_scale, downsample_scale=downsample_scale)
    return sm_np

def filter_dust(sm_np):
    sem_labels = np.unique(sm_np)
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

    pcd = o3d.geometry.PointCloud()
    xx,yy,zz =np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), np.arange(m.shape[2]), indexing='ij')
    
    pcd_np = np.vstack((xx.ravel(), yy.ravel(), zz.ravel(), m.ravel())).T

    pcd_msk = pcd_np[:,3] > 0
    pcd_np = pcd_np[pcd_msk]
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:,:3])

    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, len(mapper)))
    np.random.shuffle(color_palette)
    facecolors = color_palette[m]
    facecolors = np.reshape(facecolors, (-1, 4))[pcd_msk]
    pcd.colors = o3d.utility.Vector3dVector(facecolors[:,:3])

    # pcd, ind = pcd.remove_radius_outlier(nb_points=70, radius=5)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid], width=700, height=700)


def map_filtering(dm, sm, t1=2, downsample_scale=8):
    with torch.no_grad():
        d_msk = process_density(dm, t1=t1)
        sm_out = process_sem(sm, downsample_scale=downsample_scale, msk2=d_msk)
        sm_out = filter_dust(sm_out)

    if os.environ.get('DEBUG', False): 
        draw_map(sm_out)

    return sm_out