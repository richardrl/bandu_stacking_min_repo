import numpy as np
import torch
import open3d as o3d


def make_point_cloud_o3d(points, color):
    if isinstance(points, torch.Tensor):
        points = points.data.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    color = np.array(color)
    if len(color.shape) == 1:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    else:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd