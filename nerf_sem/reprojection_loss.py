import torch
import numpy as np
hfov = float(90) * np.pi / 180.

def depth2points(depth, extrinsic_info):
    H, W = depth.shape
    fl_x = 320
    fl_y = 320

    cx=320
    cy=240

    # xs, ys = np.meshgrid(np.arange(W), np.arange(H-1,-1,-1))
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)
    depth = depth.reshape(1,H,W)
    
    xs = (xs - cx) / fl_x
    ys = (ys - cy) / fl_y

    # this coordinate only works for Nerf, it seems has change the direction of last two dimention, so no -y -z not required
    # NOTE our raw input data should use -y -z in this step, but nerf coord can use x, y, z
    xys = np.vstack((xs * depth , ys * depth, depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = xys

    T_world_camera0 = np.array(extrinsic_info)

    # Finally transform actual points
    pcd = np.matmul(T_world_camera0, xy_c0)
    return np.transpose(pcd)[:,:3]


def get_reprojection_coord(pix_pts, depth, intrinsic, pose_before, pose_after):
    """
    Arguments:
        pix_pts: [num_pts, 3]
    """
    mapped_pts = np.zeros_like(pix_pts)
    # T = pose_after @ np.linalg.inv(pose_before)
    T = np.linalg.inv(pose_after) @ pose_before

    K = np.array([
        [intrinsic[0], 0, intrinsic[2]],
        [0, intrinsic[1], intrinsic[3]],
        [0,0,1]
    ])

    K_inv = np.array([
        [1/intrinsic[0], 0, -intrinsic[2]/intrinsic[0]],
        [0, 1/intrinsic[1], -intrinsic[3]/intrinsic[1]],
        [0,0,1]
    ])

    depth_flattened  = depth.flatten()
    ptx_3d = K_inv @ np.transpose(pix_pts)
    ptx_3d[0] *= depth_flattened
    ptx_3d[1] *= depth_flattened
    ptx_3d[2] *= depth_flattened
    ptx_3d = np.vstack([ptx_3d, np.ones_like(ptx_3d[2])])

    mapped_pts = T @ ptx_3d
    mapped_pts = K @ mapped_pts[:3, :]
    mapped_pts = np.round(np.transpose(mapped_pts[:2, :]/mapped_pts[2, :])).astype(int)
    return mapped_pts


def reprojection_loss(pix_pts, depth, intrinsic, pose1, pose2, depth_pose2, rgb = None):
    """
    Arguments:
        pix_pts: [num_pts, 2]
    """
    pix_pts = np.hstack([pix_pts, np.ones_like(pix_pts[:, 0:1])]) 
    mapped_pts = get_reprojection_coord(pix_pts, depth, intrinsic, pose1, pose2)

    projected_rgb = np.zeros_like(rgb)
    H, W = depth.shape
    msk = (mapped_pts[:, 0] >= 0) & (mapped_pts[:, 0] < W) & (mapped_pts[:, 1] >= 0) & (mapped_pts[:, 1] < H)
    projected_rgb[mapped_pts[msk][:,1], mapped_pts[msk][:,0]] = rgb[pix_pts[msk][:,1], pix_pts[msk][:,0]]

    return projected_rgb

def test_reprojection():
    pass