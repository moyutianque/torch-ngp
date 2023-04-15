import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

K = np.array([
    [400, 0, 400],
    [0, 400, 300],
    [0,0,1]
])

input_root = '../../data/replica_dinning_room/depth'
output_root = '../../data/replica_dinning_room/normal'

def get_surface_normal_by_depth(depth, K=None):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    msk = depth == 0
    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit, msk

# depth_map = np.load('../../data/replica_dinning_room/depth/31.npy')
# depth_map = depth_map/1000
# norm_map, msk = get_surface_normal_by_depth(depth_map, K) # normal vector each dim range [-1, 1]
# norm_map += 1
# norm_map /= 2
# print(np.amax(norm_map), np.amin(norm_map))
# norm_map[msk] = 0 # zero out invalid area
# plt.imshow(norm_map)
# plt.show()

for file in os.listdir(input_root):
    if file.endswith(".npy"):
        depth_map = np.load(os.path.join(input_root, file))
        depth_map = depth_map/1000

        norm_map, msk = get_surface_normal_by_depth(depth_map, K) # normal vector each dim range [-1, 1]
        norm_map[msk] = 0 # zero out invalid area

        np.save(os.path.join(output_root, file.split('.')[0]+'.npy'), {"normal": norm_map, 'msk':msk})