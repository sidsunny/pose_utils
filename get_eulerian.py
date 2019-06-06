import numpy as np
import scipy.io as sio

from transforms3d.euler import EulerFuncs
euler_fx = EulerFuncs('rxyz')

limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]
joint_names = [
    'pelvis', 'neck',  # 2
    'right_shoulder', 'right_elbow', 'right_wrist',  # 5
    'left_shoulder', 'left_elbow', 'left_wrist',  # 8
    'head_top',  # 9
    'right_hip', 'right_knee', 'right_ankle', 'right_foot',  # 13
    'left_hip', 'left_knee', 'left_ankle', 'left_foot'  # 17
]

joint_names = np.array(joint_names)
n_joints = len(limb_parents)
joint_map = {joint_names[i]: i for i in range(n_joints)}
def get_joint_index(joint_name):
    return joint_map.get(joint_name, -1)
default_coordinate_transform = 'azimuth'

def mat2euler(T):
    return euler_fx.mat2euler(T)

def unit_norm(v, axis=0):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + 1e-9)

def get_eulerian_local_coordinates(skeleton):
    right_hip = get_joint_index('right_hip')
    left_hip = get_joint_index('left_hip')
    neck = get_joint_index('neck')

    r = skeleton[right_hip]
    l = skeleton[left_hip]
    n = skeleton[neck]

    m = 0.5 * (r + l)
    z_ = unit_norm(n - m)
    y_ = unit_norm(np.cross(l - r, n - r))
    x_ = np.cross(y_, z_)

    return x_, y_, z_

def get_local_coordinates(skeleton, system=default_coordinate_transform):
    if system == 'euler':
        return get_eulerian_local_coordinates(skeleton)
    if system == 'azimuth':
        return get_azimuthal_local_coordinates(skeleton)
    raise NotImplementedError('Invalid system: %s' % system)
    
def get_global_angles_and_transform(skeleton):
    x_, y_, z_ = get_local_coordinates(skeleton, 'euler')

    B = np.matrix([x_, y_, z_])

    alpha, beta, gamma = mat2euler(B)

    return alpha, beta, gamma, np.array(skeleton * B.T)

def root_relative_to_view_norm_skeleton(skeleton):
    try:
        if np.all(skeleton == 0.):
            return np.array([0., 0., 0.]), skeleton.copy(), skeleton.copy()
        alpha, beta, gamma, view_norm_skeleton = get_global_angles_and_transform(skeleton)
        return np.array([alpha, beta, gamma]), view_norm_skeleton
    except Exception as ex:
        print ex, 'Error while processing: '
        print skeleton


pose_path_3d = '../mads_parsed/parsed_data/poses_3d/HipHop_HipHop1_C0.mat'
poses_3d = sio.loadmat(pose_path_3d)['pose_3d']
[alpha, beta, gamma], transformed_pose_3d = root_relative_to_view_norm_skeleton(poses_3d[0])

skeleton_images = skeleton_to_image(np.expand_dims(transformed_pose_3d, 0))
skeleton_images = skeleton_to_image(np.expand_dims(pred_ske_3d,0))