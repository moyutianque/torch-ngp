"""
transforms.json

fl_x, fl_y are camera intrinsic focal length x,y
cx, cy are camera intrinsic center point x,y
k1, k2, p1, p2 are camera distortion parameters
"""
import cv2
import math
import os
import os.path as osp
import numpy as np
import json

## SCANNET 0031
# fl_x=1180.313110
# fl_y=1180.313110
# cx=647.750000
# cy=483.750000

fl_x=1165.484009
fl_y=1164.543945
cx=654.941589
cy=477.277008
w=1296
h=968

## Replica
fl_x=320
fl_y=240
cx=319.5
cy=239.5
w=640
h=480

## Replica
fl_x=400
fl_y=300
cx=399.5
cy=299.5
w=800
h=600


SEM=False
AABB_SCALE=2
camera_scale_factor=2 # scale the camera position by factor
camera_angle_x = math.atan(w / (fl_x * 2)) * 2
camera_angle_y = math.atan(h / (fl_y * 2)) * 2

k1=k2=p1=p2=0.0

out = {
    "camera_angle_x": camera_angle_x,
    "camera_angle_y": camera_angle_y,
    "fl_x": fl_x,
    "fl_y": fl_y,
    "k1": k1,
    "k2": k2,
    "p1": p1,
    "p2": p2,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
    "aabb_scale": AABB_SCALE,
    "frames": [],
}


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def read_pose(pose_path):
    with open(pose_path, "r") as f:
        lines = f.readlines()
    ls = []
    for line in lines:
        l = list(map(float, line.split(' ')))
        ls.append(l)
    c2w = np.array(ls).reshape(4, 4)
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return c2w

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def post_processing(up=None):
    """ Post processing c2w to centered in a unit cube """
    nframes = len(out["frames"])

	# don't keep colmap coords - reorient the scene to be easier to work with
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    p1, p2 = np.copy(out["frames"][0]["transform_matrix"][0:3,3]), np.copy(out["frames"][200]["transform_matrix"][0:3,3])
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] *= camera_scale_factor / avglen # scale to "nerf sized"
    p11, p21 = out["frames"][0]["transform_matrix"][0:3,3], out["frames"][200]["transform_matrix"][0:3,3]
    
    # Verify scaling factor
    from scipy.spatial.distance import euclidean
    d1 = euclidean(p1,p2)
    d2 = euclidean(p11, p21)
    print("distance scale factor real: ", d2/d1)

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")

    scale_factor =  camera_scale_factor / avglen
    out['scale_factor'] = scale_factor


def process_scannet(input_root, step_size=10):
    if SEM:
        rgb_path = osp.join(input_root, '../semseg2')
    else:
        rgb_path = osp.join(input_root, 'color')

    gt_c2w_path = osp.join(input_root, 'pose')
    tot_num = len(os.listdir(rgb_path))
    up = np.zeros(3)
    for i in range(0,tot_num,step_size):
        if SEM:
            im_name = osp.join(rgb_path, f"{i}.png")
            name=f'sem/{i}.png'
        else:
            im_name = osp.join(rgb_path, f"{i}.jpg")
            name=f'color/{i}.jpg'

        c2w_name=f'{i}.txt'
        b=sharpness(im_name)
        print(name, "sharpness=",b)
        c2w = read_pose(osp.join(gt_c2w_path, c2w_name))
        # c2w = np.linalg.inv(m)
        frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
        out["frames"].append(frame)

        up += c2w[0:3,1]

    post_processing(up)

# extra_xf = np.matrix([
#             [-1, 0, 0, 0],
#             [ 0, 0, 1, 0],
#             [ 0, 1, 0, 0],
#             [ 0, 0, 0, 1]])
# # NerF will cycle forward, so lets cycle backward.
# shift_coords = np.matrix([
#     [0, 0, 1, 0],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]])

def Rx(theta):
    return np.matrix([[ 1, 0            , 0            ,0],
                    [ 0, np.cos(theta),-np.sin(theta),0],
                    [ 0, np.sin(theta), np.cos(theta),0],
                    [0,0,0,1]])
def Ry(theta):
    return np.matrix([[ np.cos(theta), 0, np.sin(theta),0],
                    [ 0            , 1, 0            ,0],
                    [-np.sin(theta), 0, np.cos(theta),0],
                    [0,0,0,1]])
def Rz(theta):
    return np.matrix([[ np.cos(theta), -np.sin(theta), 0 , 0],
                    [ np.sin(theta), np.cos(theta) , 0 , 0],
                    [ 0            , 0             , 1 , 1],
                    [0,0,0,1]])

def transfer_back(m):
    T = np.array([[1, 0, 0, 0],
                [0, np.cos(np.pi), -np.sin(np.pi), 0],
                [0, np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 0, 1]])
    return m @ np.linalg.inv(T)

def generate_transform_matrix(c2w):
    # https://github.com/NVlabs/instant-ngp/issues/72
    pos = c2w[0:3, 3]
    xf_rot = np.eye(4)
    xf_rot[:3,:3] = c2w[:3,:3]

    xf_pos = np.eye(4)
    # xf_pos[:3,3] = pos - average_position
    xf_pos[:3,3] = pos

    # barbershop_mirros_hd_dense:
    # - camera plane is y+z plane, meaning: constant x-values
    # - cameras look to +x

    # Don't ask me...
    extra_xf = np.matrix([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    # NerF will cycle forward, so lets cycle backward.
    shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    xf = shift_coords @ extra_xf @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    xf = np.asarray(xf)
    return xf

def process_replica(input_root, step_size=10, select_index=None):
    rgb_path = osp.join(input_root, 'color')
    traj_file = osp.join(input_root, 'traj.txt')

    tot_num = len(os.listdir(rgb_path))
    up = np.zeros(3)

    # load gt poses
    poses = np.loadtxt(traj_file).reshape(-1, 4, 4)

    for i in range(0,tot_num,step_size):
        if i % step_size > 0 and select_index is None:
            continue

        if (select_index is not None) and (i not in select_index):
            continue
        # im_name = osp.join(rgb_path, f"rgb_{i}.png")
        # name=f'rgb/rgb_{i}.png'
        im_name = osp.join(rgb_path, f"{i}.png")
        name=f'color/{i}.png'
        sem_name=f'sem/{i}.png'

        b=sharpness(im_name)
        print(name, "sharpness=",b)
        c2w = poses[i]
        # c2w = generate_transform_matrix(c2w)
        # TODO not known whether helpful
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = c2w[[1,0,2,3],:] # swap y and z
        c2w[2,:] *= -1

        # c2w = np.linalg.inv(m)
        frame={"file_path":name, "sem_path":sem_name, "sharpness":b,"transform_matrix": c2w}
        out["frames"].append(frame)

        up += c2w[0:3,1]

    post_processing(up)

def dump_file(dump_path):
    print(f"writing {dump_path}")
    with open(dump_path, "w") as outfile:
        json.dump(out, outfile, indent=2)

if __name__=='__main__':
    # input_root='/home/zeke/ProjectUbuntu/Projects/instant-ngp/Datasets/ScanNet/scans/scene0031_00/frames'
    # dump_path = '../outputs/scannet0031/transforms.json'
    # process_scannet(input_root, step_size=10)
    # dump_file(dump_path)

    # input_root='./outputs/replica_multi_room'
    # dump_path = './outputs/replica_multi_room/transforms.json'
    select_index = None
    # input_root='data/replica_apart2_instance'
    # dump_path = './data/replica_apart2_instance/transforms.json'

    # input_root='data/replica_apart2_dinning_room'
    # dump_path = './data/replica_apart2_dinning_room/transforms.json'
    
    input_root='test_output'
    dump_path='test_output/transforms.json'
    
    # input_root='./outputs/replica_singleroom'
    # dump_path = './outputs/replica_singleroom/transforms.json'
    # input_root='./outputs/test_sim'
    # dump_path = './outputs/test_sim/transforms.json'

    # input_root='./outputs/test_sim_multiview'
    # dump_path = './outputs/test_sim_multiview/transforms.json'
    process_replica(input_root, step_size=1, select_index=select_index)
    dump_file(dump_path)
