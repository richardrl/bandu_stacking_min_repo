import numpy as np

def fix_normal_orientation(triangles_centers, normals, com):
    assert len(normals.shape) == 2
    out_normals = []

    for idx, normal in enumerate(normals):
        ray_to_com = triangles_centers[idx] - com
        # bandu_logger.debug("mesh_util 157")
        # bandu_logger.debug(np.dot(ray_to_com, normal) < 0)
        if np.dot(ray_to_com, normal) > 0:
            out_normals.append(normal)
        else:
            out_normals.append(-normal)
    return np.array(out_normals)