import open3d as o3d
import jsonlines
import os.path as osp
import numpy as np
import json
from habitat_sim.utils.common import quat_to_magnum
import quaternion

hfov = float(90) * np.pi / 180.

def depth2points(depth_im, extrinsic_info, mode='plane'):
    fl_x=320
    fl_y=240
    cx=319.5
    cy=239.5

    H, W = depth_im.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H-1,-1,-1))
    depth = depth_im.reshape(1,H,W)/1000
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)

    xs = (xs - fl_x) / cx
    ys = (ys - fl_y) / cy

    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    msk = depth > 0
    msk = msk.flatten()
    xys = xys.reshape(4, -1)
    xy_c0 = xys[:, msk]

    translation = np.array(extrinsic_info['habitat_cam_pos']['position'])
    orientation = quaternion.from_float_array(np.array(extrinsic_info['habitat_cam_pos']['rotation']))
    rotation_0 = quaternion.as_rotation_matrix(orientation)
    T_world_camera0 = np.eye(4)
    T_world_camera0[0:3,0:3] = rotation_0
    T_world_camera0[0:3,3] = translation

    # Finally transform actual points
    pcd = np.matmul(T_world_camera0, xy_c0)
    return np.transpose(pcd)[:,:3]


def run(depth_root):
    file_paths = []
    cam_poses = []
    cam_poses = []
    with jsonlines.open(osp.join(depth_root, 'traj.jsonl')) as reader:
        for i, line in enumerate(reader):
            cam_poses.append(line)
            file_path = f"depth/{i}.npy"
            file_paths.append(file_path)
    
    # file_paths = [file_paths[0], file_paths[40],]
    # cam_poses = [cam_poses[0], cam_poses[40], ]
    # file_paths = [file_paths[64], file_paths[65]]
    # cam_poses = [cam_poses[64],  cam_poses[65]]
    # file_paths = [file_paths[63], file_paths[97]]
    # cam_poses = [cam_poses[63], cam_poses[97]]
    # file_paths = file_paths[::80]
    # cam_poses = cam_poses[::80]
    # file_paths = [file_paths[0], file_paths[1]]
    # cam_poses = [cam_poses[0], cam_poses[1]]

    pcd_nps = []
    for file_name, cam_pose in zip(file_paths, cam_poses):
        depth_im = np.load(osp.join(depth_root, file_name))
        pcd_np = depth2points(depth_im, extrinsic_info=cam_pose, mode='pinhole')
        pcd_nps.append(pcd_np)
    
    pcd_nps = np.vstack(pcd_nps) # color become more red while z value increase
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_nps)
    o3d.visualization.draw_geometries([pcd_o3d])

if __name__ == '__main__':
    depth_root = '/media/zeke/project_data/Projects/torch-ngp/my_local_process/run_sim/test_output'
    # depth_root = '/media/zeke/project_data/Projects/torch-ngp/my_local_process/run_sim/test_output-bk'
    run(depth_root)