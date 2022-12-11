import torch
import copy

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


