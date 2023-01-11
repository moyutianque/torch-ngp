import open3d as o3d
import jsonlines
import os.path as osp
import numpy as np
import json
from habitat_sim.utils.common import quat_to_magnum
import quaternion
import cv2
import matplotlib.pyplot as plt

hfov = float(90) * np.pi / 180.

def depth2points(depth_im, extrinsic_info, color_im=None):
    H, W = depth_im.shape
    vfov = 2 * np.arctan(np.tan(hfov/2)*H/W)
    fl_x = W / (2 * np.tan(hfov / 2.)) # 320
    fl_y = H / (2 * np.tan(vfov / 2.)) # 320

    cx=320
    cy=240

    # xs, ys = np.meshgrid(np.arange(W), np.arange(H-1,-1,-1))
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    depth = depth_im.reshape(1,H,W)/1000
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)
    
    xs = (xs - cx) / fl_x
    ys = (ys - cy) / fl_y

    xys = np.vstack((xs * depth , -ys * depth, -depth, np.ones(depth.shape)))
    msk = depth > 0
    msk = msk.flatten()
    xys = xys.reshape(4, -1)
    xy_c0 = xys[:, msk]

    colors = None
    if color_im is not None:
        colors = color_im[:,:,:3]
        colors = colors.reshape(-1, 3)
        colors = colors[msk, :]

    T_world_camera0 = np.array(extrinsic_info)

    # Finally transform actual points
    pcd = np.matmul(T_world_camera0, xy_c0)
    return np.transpose(pcd)[:,:3], colors

def run(depth_root):
    file_paths = []
    rgb_paths = []
    cam_poses = []
    cam_poses = []
    meta = json.load(open(osp.join(depth_root, 'transforms.json')))
    scale_factor = meta['scale_factor']

    for line in meta['frames']:
        i = int(line['file_path'].split('/')[-1].split('.')[0])
        file_path = f"depth/{i}.npy"
        file_paths.append(file_path)
        rgb_paths.append(f"color/{i}.png")
        cam_poses.append(line["transform_matrix"])
    
    # file_paths = [file_paths[0], file_paths[40],]
    # cam_poses = [cam_poses[0], cam_poses[40], ]
    # file_paths = [file_paths[64], file_paths[65]]
    # cam_poses = [cam_poses[64],  cam_poses[65]]
    # rgb_paths = [rgb_paths[64],  rgb_paths[65]]
    
    # file_paths = [file_paths[63], file_paths[97]]
    # cam_poses = [cam_poses[63], cam_poses[97]]
    file_paths = file_paths[::20]
    cam_poses = cam_poses[::20]
    rgb_paths = rgb_paths[::20]
    # file_paths = [file_paths[0], file_paths[1]]
    # cam_poses = [cam_poses[0], cam_poses[1]]

    pcd_nps = []
    color_list = []
    for file_name, rgb_path, cam_pose in zip(file_paths, rgb_paths, cam_poses):
        depth_im = np.load(osp.join(depth_root, file_name)) * scale_factor
        image = cv2.imread(osp.join(depth_root, rgb_path), cv2.IMREAD_UNCHANGED)
        if image.shape[-1] == 3: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        pcd_np, colors = depth2points(depth_im, extrinsic_info=cam_pose, color_im=image)
        color_list.append(colors)
        pcd_nps.append(pcd_np)
    
    pcd_nps = np.vstack(pcd_nps) # color become more red while z value increase
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_nps)
    if color_list[0] is not None:
        color_list = np.vstack(color_list)
        pcd_o3d.colors = o3d.utility.Vector3dVector(color_list/255.0)
    o3d.visualization.draw_geometries([pcd_o3d])
    # o3d.io.write_point_cloud("./data.ply", pcd_o3d)


if __name__ == '__main__':
    # depth_root = '/media/zeke/project_data/Projects/torch-ngp/my_local_process/run_sim/test_output'
    depth_root = '/media/zeke/project_data/Projects/torch-ngp/data/replica_multiroom2'
    run(depth_root)