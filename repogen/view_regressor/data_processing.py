import json
import numpy as np
import torch


def c2s(pts, use_torch=False):
    if use_torch:
        x, y, z = torch.unbind(pts, dim=1)
    else:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if use_torch:
        r = torch.norm(pts, dim=1)
        theta = torch.atan2(
            torch.sqrt(x*x + y*y),
            z,
        )
        phi = torch.atan2(y, x)
        spherical = torch.stack([r, theta, phi], dim=1)
    else:
        r = np.linalg.norm(pts, axis=1)
        theta = np.arctan2(
            np.sqrt(x*x + y*y),
            z,
        )
        phi = np.arctan2(y, x)
        spherical = np.stack([r, theta, phi], axis=1)
    return spherical


def s2c(pts, use_torch=False):
    if use_torch:
        fn = torch
    else:
        fn = np

    if pts.shape[1] == 3:
        if use_torch:
            r, theta, phi = torch.unbind(pts, dim=1)
        else:
            r = pts[:, 0]
            theta = pts[:, 1]
            phi = pts[:, 2]
    else:
        if use_torch:
            theta, phi = torch.unbind(pts, dim=1)
            r = torch.ones(pts.shape[0]).to(theta.device) # If no radius given, use 1
        else:
            r = np.ones(pts.shape[0]) # If no radius given, use 1
            theta = pts[:, 0]
            phi = pts[:, 1]

    x = r * fn.sin(theta) * fn.cos(phi)
    y = r * fn.sin(theta) * fn.sin(phi)
    z = r * fn.cos(theta)
    return fn.stack([x, y, z], axis=1)


def load_data_from_coco_file(coco_filepath, views_filepath=None, remove_limbs=False, num_visible_keypoints=4):
    coco_dict = json.load(open(coco_filepath, "r"))
    image_ids = []
    keypoints = []
    bboxes_xywh = []
    
    if not views_filepath is None:
        views_dict = json.load(open(views_filepath, "r"))
        positions = []

    for annot in coco_dict["annotations"]:
        image_id = annot["image_id"]
        image_name = "{:d}.jpg".format(image_id)
        kpts = np.array(annot["keypoints"])
        bbox = np.array(annot["bbox"])

        # Remove limbs
        if remove_limbs:
            kpts = np.reshape(kpts, (-1, 3))
            kpts[[7, 8, 9, 10, 13, 14, 15, 16], :] = 0
            # kpts = kpts[[
            #     0,      # Nose
            #     1,      # Left eye
            #     2,      # Right eye
            #     3,      # Left ear
            #     4,      # Right ear
            #     5,      # Left shoulder
            #     6,      # Right shoulder
            #     11,     # Left hip
            #     12,     # Right hip
            # ], :].flatten()
            kpts = kpts.flatten()

        # At least 4 keypoints must be visible
        vis_mask = kpts[2::3] > 1
        if np.sum(vis_mask) < (num_visible_keypoints-1):
            continue

        image_ids.append(image_id)
        keypoints.append(kpts)
        bboxes_xywh.append(bbox)
        if not views_filepath is None:
            view = views_dict[image_name]
            camera_pos = view["camera_position"]
            positions.append(camera_pos)

    keypoints = np.array(keypoints)
    bboxes_xywh = np.array(bboxes_xywh)
    image_ids = np.array(image_ids).squeeze()
    
    if not views_filepath is None:
        positions = np.array(positions)
        return keypoints, bboxes_xywh, image_ids, positions

    return keypoints, bboxes_xywh, image_ids


def process_keypoints(keypoints, bboxes, add_visibility=False, add_bboxes=True, normalize=True, remove_limbs=False):
    """
    Process the keypoints to minimize the domain gap between synthetic and COCO keypoints.
    1. Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    2. Remove keypoints with visibility < 2
    """

    num_samples = keypoints.shape[0]
    num_keypoints = 17

    keypoints = np.reshape(keypoints, (-1, num_keypoints, 3)).astype(np.float32)
    
    # Normalize the keypoints to be in the range [0, 1] with respect to the bounding box
    bboxes = bboxes[:, None, :]
    if normalize:
        keypoints[:, :, 0] = (keypoints[:, :, 0] - bboxes[:, :, 0]) / bboxes[:, :, 2]
        keypoints[:, :, 1] = (keypoints[:, :, 1] - bboxes[:, :, 1]) / bboxes[:, :, 3]

    # Remove keypoints with visibility < 2
    visibilities = keypoints[:, :, 2].squeeze()
    keypoints[visibilities < 2, :] = 0

    # Remove the visibility flag from the keypoints
    if not add_visibility:
        keypoints = keypoints[:, :, :2]

    # Stack bbox width and height to the keypoints
    if add_bboxes:
        if add_visibility:
            keypoints = np.reshape((num_samples, -1))
            keypoints = np.concatenate([keypoints, bboxes[:, :, 2:].squeeze()], axis=1)
        else:
            keypoints = np.concatenate([keypoints, bboxes[:, :, 2:]], axis=1)
    
    # Reshape the keypoints to be a 1D array
    keypoints = np.reshape(keypoints, (num_samples, -1))
    print("Keypoints shape:", keypoints.shape)
    
    # for coor in range(keypoints.shape[1]):
    #     print("Coordinate {:d}: min={:.3f}, max={:.3f}".format(coor, np.min(keypoints[:, coor]), np.max(keypoints[:, coor])))
    
    return keypoints


def occlude_random_keypoints(
        keypoints,
        min_num_keypoints=8,
        has_bbox=True,
    ):
    """
    Occlude a random subset of keypoints.
    """


    keypoints = np.reshape(keypoints.copy(), (-1, 2))
    if has_bbox:
        bbox = keypoints[-1, :]
        keypoints = keypoints[:-1, :]

    valid_keypoints = keypoints[keypoints[:, 0] > 0, :]

    num_keypoints = valid_keypoints.shape[0]
    num_keypoints_to_keep = np.random.randint(min_num_keypoints, num_keypoints+1)

    # Randomly select keypoints to occlude
    occlude_mask = np.ones(num_keypoints, dtype=bool)
    occlude_mask[:num_keypoints_to_keep] = False
    np.random.shuffle(occlude_mask)
    
    valid_keypoints[occlude_mask, :] = 0
    keypoints[keypoints[:, 0] > 0, :] = valid_keypoints

    if has_bbox:
        keypoints = np.concatenate([keypoints, bbox[None, :]], axis=0)
    
    keypoints = keypoints.flatten()
    return keypoints


def occlude_keypoints_with_rectangle(
        keypoints,
        min_num_keypoints=8,
        has_bbox=True,
    ):
    """
    Occlude a random subset of keypoints with a rectangle.
    """
    keypoints = np.reshape(keypoints.copy(), (-1, 2))
    if has_bbox:
        bbox = keypoints[-1, :]
        keypoints = keypoints[:-1, :]

    num_keypoints = np.sum(keypoints[:, 0] > 0)
    num_keypoints_to_keep = np.random.randint(min_num_keypoints, num_keypoints+1)
    
    # Select a random point and occlude X nearest keypoints
    random_pt = np.random.rand(2)
    dists = np.linalg.norm(keypoints - random_pt, axis=1)
    dists[keypoints[:, 0] <= 0] = np.inf
    nearest_keypoints = np.argsort(dists)[num_keypoints_to_keep:]
    keypoints[nearest_keypoints, :] = 0

    if has_bbox:
        keypoints = np.concatenate([keypoints, bbox[None, :]], axis=0)
    
    keypoints = keypoints.flatten()
    return keypoints


def randomly_occlude_keypoints(keypoints, has_bbox=True, min_num_keypoints=6, rectangle_pst=0.3, occlusion_pst=0.1):
    """
    Randomly occlude keypoints with either a rectangle or random points.
    """
    is_pytorch = isinstance(keypoints, torch.Tensor)
    if is_pytorch:
        keypoints = keypoints.cpu().numpy()

    keypoints = keypoints.copy()
    has_batches = len(keypoints.shape) == 2

    if has_batches:
        num_samples = keypoints.shape[0]
        for i in range(num_samples):
            keypoints[i, :] = randomly_occlude_keypoints(
                keypoints[i, :].squeeze(),
                has_bbox=has_bbox,
                min_num_keypoints=min_num_keypoints,
                rectangle_pst=rectangle_pst,
                occlusion_pst=occlusion_pst,
            )
    else:
        keypoints = np.reshape(keypoints, (-1, 2))

        if has_bbox:
            num_valid_keypoints = np.sum(keypoints[:-1, 0] > 0)
        else:
            num_valid_keypoints = np.sum(keypoints[:, 0] > 0)
        
        if num_valid_keypoints > min_num_keypoints:
            random_action = np.random.choice(
                ["rectangle", "occlusion", None],
                p=[rectangle_pst, occlusion_pst, 1-(rectangle_pst+occlusion_pst)],
            )

            if random_action == "rectangle":
                keypoints = occlude_keypoints_with_rectangle(keypoints, min_num_keypoints=min_num_keypoints, has_bbox=has_bbox)
            elif random_action == "occlusion":
                keypoints = occlude_random_keypoints(keypoints, min_num_keypoints=min_num_keypoints, has_bbox=has_bbox)

        keypoints = keypoints.flatten()

    if is_pytorch:
        keypoints = torch.from_numpy(keypoints)
    
    return keypoints


def angular_distance(pts1, pts2, use_torch=False):
    """
    Compute the angular distance between two points on a unit sphere.
    """
    if use_torch:
        acos = torch.arccos
        sin = torch.sin
        cos = torch.cos
        all = torch.all
        any = torch.any
        abs = torch.abs
        clip = torch.clamp
    else:
        acos = np.arccos
        sin = np.sin
        cos = np.cos
        all = np.all
        any = np.any
        abs = np.abs
        clip = np.clip

    if pts1.shape[1] == 3:
        radius1, theta1, phi1 = pts1[:, :1], pts1[:, 1:2], pts1[:, 2:]
        radius2, theta2, phi2 = pts2[:, :1], pts2[:, 1:2], pts2[:, 2:]
    else:
        theta1, phi1 = pts1[:, :1], pts1[:, 1:2]
        theta2, phi2 = pts2[:, :1], pts2[:, 1:2]

    # Clip the input - not sure if this helped any
    # theta1 = clip(theta1, 0, np.pi)
    # theta2 = clip(theta2, 0, np.pi)
    # phi1 = clip(phi1, -np.pi, np.pi)
    # phi2 = clip(phi2, -np.pi, np.pi)
        
    dist = acos(sin(theta1)*sin(theta2) + cos(theta1)*cos(theta2)*cos(phi1 - phi2))

    # Add radius difference - not sure if this helped any
    # if pts1.shape[1] == 3:
    #     dist += 1.0 * abs(radius1 - radius2)

    # if not all(dist >= 0):
    #     print(pts1[:10])
    #     print(pts2[:10])
    #     print(dist[:10])

    assert all(dist >= 0)
    # assert all(dist <= np.pi)

    return dist


if __name__ == "__main__":

    pts1 = np.array([
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [1, 1, 0],
        [1, 0, -1],
    ], dtype=np.float32)
    pts2 = np.zeros(pts1.shape, dtype=np.float32)
    pts2[:, -1] = 1
    d = angular_distance(c2s(pts1), c2s(pts2))
    d = d * 180 / np.pi
    
    print("Distance of selected points on a unit sphere (in degrees):")
    for i in range(pts1.shape[0]):
        print("Points {} x {}:\t\t{:.3f}".format(pts1[i, :], pts2[i, :], d[i]))
    
    pts1 = np.random.normal(size = (100000, 3))
    pts2 = np.random.normal(size = (100000, 3))
    d = angular_distance(c2s(pts1), c2s(pts2))
    d = d * 180 / np.pi

    print()
    print("Distance of random points on a unit sphere (in degrees):")
    print("Min: {:.3f}, Mean: {:.3f}, Max: {:.3f}".format(np.min(d), np.mean(d), np.max(d)))