import torch
import copy
import numpy as np

def zero_pads(data, voxel_size =16):
    # if data.size(0)==1:
    #     data = data.squeeze(0)

    size = list(data.size())

    new_size = copy.deepcopy(size)
    for i in range(1,len(size)):
        if new_size[i]%voxel_size==0:
            continue
        new_size[i] = (new_size[i]//voxel_size+1)*voxel_size
    
    res= torch.zeros(new_size, device = data.device)
    res[:,:size[1],:size[2],:size[3]] = data.clone()
    return res

def zero_pads_np(data, voxel_size =16):
    size = list(data.shape)[:3]

    new_size = copy.deepcopy(size)
    for i in range(1,len(size)):
        if new_size[i]%voxel_size==0:
            continue
        new_size[i] = (new_size[i]//voxel_size+1)*voxel_size
    
    data = np.pad(data, pad_width=((0, new_size[0]-size[0]), (0, new_size[1]-size[1]), (0, new_size[2]-size[2]), (0,0)))
    return data

def zero_unpads(data, size):
    if len(size) == 4:
        size = size[1:]
    return data[:,:size[0],:size[1],:size[2]]


def split_volume(data, voxel_size = 16):
    size = list(data.size())
    for i in range(1,len(size)):
        size[i] = size[i]//voxel_size

    res = []
    for x in range(size[1]):
        for y in range(size[2]):
            for z in range(size[3]):
                res.append(data[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size].clone())

    res = torch.stack(res)
    return res,size[1:]


def merge_volume(data, size):
    voxel_size = data.size(-1)
    res=torch.zeros(data.size(1), size[0]*voxel_size,size[1]*voxel_size, size[2]*voxel_size,device = data.device)
    cnt = 0
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                res[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size] = data[cnt,:,:,:,:]
                cnt = cnt + 1

    return res


import numpy as np
import skimage
import multiprocessing
from scipy.stats import mode
from scipy.stats import entropy

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

def entropy1(labels, base=None):
    value, counts = np.unique(labels[labels!=0], return_counts=True)
    ep_val = entropy(counts, base=base)
    # if len(counts)>0 and np.amax(counts/len(labels)) > 0.25:
    #     return ep_val
    # return 1.
    return ep_val

def entropy_filter(np_arr, kernel_size, threshold=0.1):
    pad_size = int(kernel_size//2)
    np_arr_chunks = skimage.util.view_as_windows(np.pad(np_arr, (pad_size,pad_size), mode='constant'), window_shape=(kernel_size, kernel_size, kernel_size))
    np_arr_chunks = np.reshape(np_arr_chunks, (int(np.prod(np_arr_chunks.shape[:3])), -1))
    out = parallel_apply_along_axis(entropy1, axis=1, arr=np_arr_chunks)
    return out < threshold

def mode_cus(labels, threshold=0.25):
    vals, cnts = np.unique(labels, return_counts=True)

    if vals[0] == 0:
        if len(vals) == 1:
            return 0
        cnts = cnts[1:]
        vals = vals[1:]
    
    ep_val = entropy(cnts, base=None)
    if ep_val > 0.1:
        return 0
    else:
        if cnts[0]/len(labels) > (threshold/4):
            return vals[0]
        else:
            return 0

    # if vals[0] == 0 and len(vals) == 1:
    #     return 0

    # if vals[0] == 0:
    #     if cnts[1]/len(labels) >= threshold:
    #         return vals[1]
    # else:
    #     if cnts[0]/len(labels) >= threshold:
    #         return vals[0]
    # return 0

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
