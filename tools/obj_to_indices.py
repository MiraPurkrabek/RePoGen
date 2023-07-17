"""
This script takes an OBJ file and a target mesh (in NPZ format) and outputs
a vector of indices that can be used to map the OBJ file to the target mesh.
Used for mapping SMPL-X UVs to the SMPL-X mesh.
"""

import numpy as np
import trimesh
import scipy
import argparse

import pyrender

from examples.sample_random_poses import random_camera_pose

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='models/smplx/smplx_uv.obj')
    parser.add_argument('--output_path', type=str, default='images/textures/SMPLX_MALE_indices.npz')
    parser.add_argument('--show',
                        action="store_true", default=False,
                        help='If True, will render and show results instead of saving images')
    parser.add_argument('--center',
                        action="store_true", default=False,
                        help='If True, will center the mesh accoridng to the target mesh')
    return parser.parse_args()

def main(args):


    mesh = trimesh.load(args.obj_path)

    target_npz = np.load("models/smplx/SMPLX_MALE.npz")
    orig_vertices = np.load("images/textures/vertices_generated.npz")['vertices']
    target_mesh = trimesh.Trimesh(vertices=orig_vertices, faces=target_npz["f"])

    # Center the mesh if necessary
    if args.center:
        mesh.vertices -= mesh.centroid
        target_mesh.vertices -= target_mesh.centroid

    # Compute distance matrix
    distance_matrix = scipy.spatial.distance.cdist(
        target_mesh.vertices,
        mesh.vertices
    )

    # Check distance matrix shape
    assert distance_matrix.shape[0] == len(target_mesh.vertices)
    assert distance_matrix.shape[1] == len(mesh.vertices)
    print("Distance matrix shape OK ({} x {})".format(*distance_matrix.shape))

    if args.show:
        scene = pyrender.Scene(bg_color=(255, 255, 255, 0))

        pymesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pymesh)

        pymesh = pyrender.Mesh.from_trimesh(target_mesh, wireframe=True)
        scene.add(pymesh)

        # Add lights
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=5e2)
        for _ in range(5):
            scene.add(light, pose=random_camera_pose(distance=3))

        pyrender.Viewer(scene, use_raymond_lighting=True)
    
    # Find closest vertex for each vertex
    indices_to_save = np.argmin(distance_matrix, axis=0)

    # Check indices shape
    assert (indices_to_save.shape[0] == len(mesh.vertices))
    print("Indices vector shape OK ({} x 1)".format(*indices_to_save.shape))

    assert np.min(indices_to_save) == 0
    assert np.max(indices_to_save) == len(target_mesh.vertices) - 1
    assert np.unique(indices_to_save).shape[0] == len(target_mesh.vertices)
    print("Indices vector content OK ({} unique values from {} to {})".format(
        np.unique(indices_to_save).shape[0],
        np.min(indices_to_save),
        np.max(indices_to_save)
    ))

    np.savez(args.output_path, indices=indices_to_save)
    print("Indices saved to '{}'".format(args.output_path))


if __name__ == '__main__':
    args = parse_args()
    main(args)