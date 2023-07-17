import numpy as np
import torch

def generate_pose(typical_pose=None, simplicity=5, extreme_poses=False):
    # Front rotation
    # Counter-clockwise rotation
    # Side bend

    if simplicity < 0:
        simplicity = rnd(1.0, 1.5)

    joints = {
        "Left leg": 0,
        "Right leg": 1,
        "Torso": 2,
        "Left knee": 3,
        "Right knee": 4,
        "Mid-torso": 5,
        "Left ankle": 6,
        "Right ankle": 7,
        "Chest": 8,
        "Left leg fingers": 9,
        "Right leg fingers": 10,
        "Neck": 11,
        "Left neck": 12,
        "Right neck": 13,
        "Upper neck": 14,
        "Left shoulder": 15,
        "Right shoulder": 16,
        "Left elbow": 17,
        "Right elbow": 18,
        "Left wrist": 19,
        "Right wrist": 20,
    }
    body_pose = torch.zeros((1, len(joints)*3))

    limits_deg = {
        "Left leg": [
            (-90, 45), (-50, 90), (-30, 90),
        ],
        "Right leg": [
            (-90, 45), (-90, 50), (-90, 30),
        ],
        "Torso": [
            (0, 0), (-40, 40), (-20, 20),
        ],
        "Left knee": [
            (0, 150), (0, 0), (0, 0)
        ],
        "Right knee": [
            (0, 150), (0, 0), (0, 0)
        ],
        "Mid-torso": [
            (0, 20), (-30, 30), (-30, 30)
        ],
        "Left ankle": [
            (-20, 70), (-15, 5), (0, 20)
        ],
        "Right ankle": [
            (-20, 70), (-5, 15), (-20, 0)
        ],
        "Chest": [
            (0, 0), (-20, 20), (-10, 10),
        ],
        "Left leg fingers": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Right leg fingers": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Neck": [
            (-45, 45), (-40, 40), (-20, 20)
        ],
        "Left neck": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Right neck": [
            (0, 0), (0, 0), (0, 0)
        ],
        "Upper neck": [
            (-20, 20), (-30, 30), (-10, 10)
        ],
        "Left shoulder": [
            # Changed to move 0 along the body
            (-30, 30), (-90, 20), (-10, 140)
        ],
        "Right shoulder": [
            # Changed to move 0 along the body
            (-30, 30), (-20, 90), (-140, 10)
        ],
        "Left elbow": [
            (0, 0), (-120, 0), (0, 0)
        ],
        "Right elbow": [
            (0, 0), (0, 120), (0, 0)
        ],
        "Left wrist": [
            (-30, 30), (-10, 10), (-70, 90)
        ],
        "Right wrist": [
            (-30, 30), (-10, 10), (-90, 70)
        ],
    }
    
    if typical_pose is None:
        # Generate a completely random pose
        min_limits = []
        max_limits = []

        for _, lims in limits_deg.items():
            if len(lims) > 0:
                for l in lims:
                    min_limits.append(l[0])
                    max_limits.append(l[1])
            else:
                min_limits += [0, 0, 0]
                max_limits += [0, 0, 0]

        min_limits = torch.Tensor(min_limits) / simplicity
        max_limits = torch.Tensor(max_limits) / simplicity

        if extreme_poses:
            # Sample poses uniformly from the limits - much more extreme poses
            joints_rng = max_limits - min_limits
            random_angles = torch.rand((len(joints)*3)) * joints_rng + min_limits
        else:        
            # Generate random angles
            random_angles = torch.normal(
                mean = torch.zeros((len(joints)*3)),
                std = 0.5,     # Experimentaly; some angles are slightly above 1.0 which is OK
            )
            random_angles[random_angles >= 0] *= max_limits[random_angles >= 0]
            random_angles[random_angles < 0] *= -min_limits[random_angles < 0]

        # Arms in the standart position are too far from the body
        random_angles[joints["Left shoulder"]*3+2]  -= 70
        random_angles[joints["Right shoulder"]*3+2] += 70
        
        # Transfer to radians
        random_angles = random_angles / 180 * np.pi

        body_pose = random_angles.reshape((1, len(joints)*3))
        
    elif typical_pose.lower() == "min":
        for joint, lims in limits_deg.items():
            for li, l in enumerate(lims):
                body_pose[0, joints[joint]*3+li] = l[0] / 180 * np.pi
    
    elif typical_pose.lower() == "max":
        for joint, lims in limits_deg.items():
            for li, l in enumerate(lims):
                body_pose[0, joints[joint]*3+li] = l[1] / 180 * np.pi

    elif typical_pose.lower() == "sit":
        body_pose[0, joints["Right knee"]*3+0] = np.pi / 2
        body_pose[0, joints["Left knee"]*3+0] = np.pi / 2
        body_pose[0, joints["Right leg"]*3+0] = - np.pi / 2
        body_pose[0, joints["Left leg"]*3+0] = - np.pi / 2
        body_pose[0, joints["Left shoulder"]*3+2] = - np.pi / 2 * 4/5
        body_pose[0, joints["Right shoulder"]*3+2] = np.pi / 2 * 4/5
    
    elif typical_pose.lower() == "stretch":
        body_pose[0, joints["Torso"]*3+0] = np.pi / 2
        body_pose[0, joints["Left shoulder"]*3+1] = - np.pi / 2 * 4/5
        body_pose[0, joints["Right shoulder"]*3+1] = np.pi / 2 * 4/5
    
    elif typical_pose.lower() == "stand":
        body_pose[0, joints["Left shoulder"]*3+2] = - np.pi / 2 * 4/5
        body_pose[0, joints["Right shoulder"]*3+2] = np.pi / 2 * 4/5
        
    return body_pose


def random_3d_position_polar(distance=2.0, view_preference=None):
    
    if distance < 0:
        distance = rnd(1.5, 5.0)
        # distance = rnd(1.5, 8.0)
    
    # Noise is 20°
    noise_size = 20 / 180 * np.pi

    # ALPHA - rotation around the sides
    # BETA  - rotation around the top and bottom
    if view_preference is None:
        # alpha_mean = 0
        # alpha_range = np.pi
        # beta_mean = 0
        # beta_range = np.pi
        raise ValueError("view_preference must be specified. Use 'random_point_on_sphere' instead.")
    elif view_preference.upper() == "PERIMETER":
        alpha_mean = 0
        alpha_range = np.pi
        beta_mean = 0
        beta_range = noise_size
    elif view_preference.upper() == "FRONT":
        alpha_mean = 0
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "BACK":
        alpha_mean = np.pi
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "SIDE":
        alpha_mean = random_sgn() * np.pi/2
        alpha_range = noise_size
        beta_mean = 0
        beta_range = 3*noise_size
    elif view_preference.upper() == "TOP":
        alpha_mean = 0
        alpha_range = 0             # Because of the hotfix in random_camera_pose
        beta_mean = np.pi/2
        beta_range = 0              # Because of the hotfix in random_camera_pose
    elif view_preference.upper() == "BOTTOM":
        alpha_mean = 0
        alpha_range = 0             # Because of the hotfix in random_camera_pose
        beta_mean = - np.pi/2
        beta_range = 0              # Because of the hotfix in random_camera_pose
    elif view_preference.upper() == "TOPBOTTOM":
        alpha_mean = 0
        alpha_range = 0             # Because of the hotfix in random_camera_pose
        beta_range = 0              # Because of the hotfix in random_camera_pose
        if np.random.rand() < 0.5:
            beta_mean = np.pi/2
        else:
            beta_mean = - np.pi/2
    else:
        raise ValueError("Unknown view preference")
    
    # [ 0,  0,  1] - FRONT view
    # [ 0,  0, -1] - BACK view (do not converge)
    # [ 0,  1,  0] - TOP view
    # [ 0, -1,  0] - BOTTOM view
    # [ 1,  0,  0] - LEFT view
    # [-1,  0,  0] - RIGHT view
    alpha = alpha_mean + rnd(-alpha_range, alpha_range)
    beta = beta_mean + rnd(-beta_range, beta_range)
    
    return alpha, beta, distance


def random_point_on_sphere(distance=2.0):
    """The projection to the sphere method"""
    
    if distance < 0:
        distance = rnd(1.5, 5.0)
        # distance = rnd(1.5, 8.0)
    
    # Generate vector big enough to avoid precision error
    pt = np.random.normal(size=3)
    while np.linalg.norm(pt) < 1e-6:
        pt = np.random.normal(size=3)

    # Normalize to the sphere
    pt = pt / np.linalg.norm(pt)
    return pt * distance


def random_point_in_sphere(distance_min=0.3, distance_max=15.0):
    """The rejection method"""
    scale = distance_max - distance_min
    pt = (np.random.uniform(size=3) * scale + distance_min) * np.array([random_sgn(), random_sgn(), random_sgn()])
    while np.linalg.norm(pt) < distance_min or np.linalg.norm(pt) > distance_max:
        pt = (np.random.uniform(size=3) * scale + distance_min) * np.array([random_sgn(), random_sgn(), random_sgn()])
    return pt


def polar_to_cartesian(alpha, beta, distance):
    y = distance * np.sin(beta)
    a = distance * np.cos(beta)
    x = a * np.sin(alpha)
    z = a * np.cos(alpha)
    return np.array([x, y, z])
    

def add_noise_to_pose(pose, noise_size=60/180*np.pi):
    current_distance = np.linalg.norm(pose)
    # pose += np.random.uniform(low=-1, high=1.0, size=3) * noise_size
    pose += np.random.normal(size=3) * noise_size
    pose = pose / np.linalg.norm(pose) * current_distance
    return pose


def random_camera_pose(distance=3, view_preference=None, rotation=0, return_vectors=False):
    
    if rotation < 0:
        rotation = rnd(0, 360)

    # Convert to radians
    rotation = rotation / 180 * np.pi


    # if view_preference is None and distance<0:
    #     camera_pos = random_point_in_sphere()
    if view_preference is None:
        camera_pos = random_point_on_sphere(distance)
    else:
        alpha, beta, distance = random_3d_position_polar(distance, view_preference)
        camera_pos = polar_to_cartesian(alpha, beta, distance)
        
        # Quick fix for the noise in TOP view
        if view_preference.upper() in ["TOP", "TOPBOTTOM", "BOTTOM"]:
            camera_pos = add_noise_to_pose(camera_pos)

    # Default camera_up is head up
    camera_up = np.array([0, 1, 0], dtype=np.float32)
    # For TOP and BOTTOM, default camera_up is front
    
    if not view_preference is None and view_preference.upper() in ["TOP", "BOTTOM"]:
        camera_up = np.array([0, 1, 0], dtype=np.float32)

    center = np.array([0, 0, 0], dtype=np.float32)

    f = center - camera_pos
    f /= np.linalg.norm(f)
    
    s = np.cross(f, camera_up)
    s /= np.linalg.norm(s)

    u = np.cross(s, f)
    u /= np.linalg.norm(u)

    R = np.array([
        [s[0], u[0], -f[0]],
        [s[1], u[1], -f[1]],
        [s[2], u[2], -f[2]],
    ])

    # theta = np.random.rand() * 2*rotation - rotation
    theta = random_sgn() * rotation + random_sgn() * rnd(0, np.pi/5)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1],
    ])
    R = R @ rot_z
    camera_up = R @ camera_up

    pose = np.array([
        [R[0, 0], R[0, 1], R[0, 2], camera_pos[0]],
        [R[1, 0], R[1, 1], R[1, 2], camera_pos[1]],
        [R[2, 0], R[2, 1], R[2, 2], camera_pos[2]],
        [      0,       0,       0,    1],
    ])

    if return_vectors:
        return pose, camera_pos, camera_up
    else:
        return pose


def rnd(a_min=0, a_max=1):
    rng = a_max-a_min
    return np.random.rand() * rng + a_min


def random_sgn():
    return -1 if np.random.rand() < 0.5 else 1

