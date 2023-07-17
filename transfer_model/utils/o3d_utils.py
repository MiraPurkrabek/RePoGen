# -*- coding: utf-8 -*-

# This is a modified version of the original code from
# the SMPL-X repository. The original code is available at
# https://github.com/vchoutas/smplx. Please note that the
# original code is free to use only for
# NON-COMMERCIAL SCIENTIFIC RESEARCH PURPOSES. For more info,
# please see the LICENSE section of the README.

import open3d as o3d
import torch

Vector3d = o3d.utility.Vector3dVector
Vector3i = o3d.utility.Vector3iVector

Mesh = o3d.geometry.TriangleMesh


def np_mesh_to_o3d(vertices, faces):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    mesh = Mesh()
    mesh.vertices = Vector3d(vertices)
    mesh.triangles = Vector3i(faces)
    return mesh
