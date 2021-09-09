import numpy
import numpy as np
import open3d
import torch


def get_skew_symmetric_matrix(vec):
    return np.asarray([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0],
    ])


def get_rotation_matrix_between_vecs(target_vec, start_vec):
    # rotation with theta = the cosine angle between the two vectors
    """

    :param target_vec:
    :param start_vec:
    :return: 3x3 rotation matrix representing relative rotation from start_vec to target_vec
    """
    # the formula fails when the vectors are collinear

    target_vec = np.array(target_vec)
    start_vec = np.array(start_vec)
    assert len(target_vec.shape) == 1 or len(target_vec.shape) == 2
    assert target_vec.shape == start_vec.shape, (target_vec.shape, start_vec.shape)
    target_vec = target_vec / np.linalg.norm(target_vec, axis=-1)
    start_vec = start_vec/np.linalg.norm(start_vec, axis=-1)

    if np.all(np.isclose(start_vec, -1 * target_vec, atol=1.e-3)):
        return R.from_euler("x", np.pi).as_matrix()

    K = get_skew_symmetric_matrix(np.cross(start_vec, target_vec))
    rotation = np.eye(3,3) + K + \
               np.dot(K, K) * \
               1/(1+np.dot(start_vec, target_vec) + 1E-7)
    return rotation


def get_relative_rotation_from_binary_logits(rotated_pointcloud,
                                             binary_logits,
                                             sigmoid_threshold=.5,
                                             dir="surface_to_upright"):
    """

    :param rotated_pointcloud: num_points x 3
    :param binary_logits: num_points
    :param dir: whether we find the rotation from the surface normal to the -z, or the rotation from -z to the surface normal
    :param com: center of mass, used to determine oriented normal
    :return:
    """
    assert len(rotated_pointcloud.shape) == 2, rotated_pointcloud.shape

    if len(binary_logits.shape) == 2:
        binary_logits = binary_logits.squeeze(-1)
    assert len(binary_logits.shape) == 1, binary_logits.shape
    assert dir in ["surface_to_upright", "upright_to_surface"]
    surface_points = rotated_pointcloud[torch.sigmoid(binary_logits) < sigmoid_threshold]
    surface_pcd = open3d.geometry.PointCloud()
    surface_pcd.points = open3d.utility.Vector3dVector(surface_points.cpu().data.numpy())

    try:
        print("using this many points...")
        print(np.min([15, surface_points.shape[0]]))
        plane_model, plane_idxs = surface_pcd.segment_plane(.007, np.min([15, surface_points.shape[0]]), 1000)
        plane_normal = np.array(plane_model)[:3]
        a, b, c, d = plane_model

        # orient the plane normal away from the center of mass
        # normal should have the same sign as the vector from the center of mass to the plane origin
        oriented_normal = np.sign(-d) * plane_normal

        if dir == "surface_to_upright":
            return get_rotation_matrix_between_vecs([0, 0, -1], oriented_normal), plane_model
        else:
            return get_rotation_matrix_between_vecs(oriented_normal, [0, 0, -1]), plane_model
    except:
        print("Unable to find any points")
        return np.eye(3), [1, 1, 1, 1]

import open3d
# from bandu.utils import pb_util, vis_util, bandu_util
from scipy.spatial.transform import Rotation as R
import copy


# def get_relative_rotation_from_obb(rotated_pointcloud, vis_o3d=False):
#     obb = open3d.geometry.OrientedBoundingBox()
#     obb = obb.create_from_points(open3d.utility.Vector3dVector(rotated_pointcloud.cpu().data.numpy()))
#
#     # make sure obb.R is orthogonal
#
#     new_obb = open3d.geometry.OrientedBoundingBox(obb)
#     new_obb.color = np.array([1., 0., 0.])
#
#     obb.color = np.array([0, 1, 0])
#
#     H = np.zeros((4, 4))
#     H[:3, :3] = R.from_matrix(obb.R).inv().as_matrix()
#     H[3, 3] = 1
#
#     # aa_obb = obb.transform(H)
#     original_orthogonal_matrix = copy.deepcopy(obb.R)
#
#     # force to be rotation matrix
#     # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
#     # u, sigma, vh = np.linalg.svd(original_orthogonal_matrix)
#     #
#     #
#     # s_new = np.zeros((3, 3))
#     # np.fill_diagonal(s_new, [1, 1, np.linalg.det(u @ vh)])
#     #
#     # original_rotation = u @ s_new @ vh
#     # original_rotation = original_orthogonal_matrix
#
#      #this doesn't work because original rotation is not actually a rotation
#     # aa_obb = open3d.geometry.OrientedBoundingBox(obb).rotate(np.linalg.inv(original_rotation), np.zeros(3))
#     # aa_obb.color = np.array([0., 0., 1.])
#
#     # 8 x 3
#     aa_box_points = (np.linalg.inv(original_orthogonal_matrix) @ np.asarray(obb.get_box_points()).T).T
#     aa_longest_vector_axis_id = np.argmax(np.max(aa_box_points, axis=0))
#
#     # aa_longest_vector_axis_id = np.argmax(aa_obb.get_max_bound())
#     obb.get_box_points()
#     aa_source_vector = np.zeros(3)
#     aa_source_vector[aa_longest_vector_axis_id] = 1
#
#     # apply original rotation to aa source vector
#     oriented_source_vector = original_orthogonal_matrix @ aa_source_vector.T
#
#     # calculate rotation from oriented source vector to negative gravity vector
#     rot_to_apply = bandu_util.get_rotation_matrix_between_vecs([0, 0, -1], oriented_source_vector)
#
#     # assert np.isclose(np.linalg.det(rot_to_apply), 1, atol=.001), np.linalg.det(rot_to_apply)
#     if vis_o3d:
#         open3d.visualization.draw_geometries([obb,
#                                               # aa_obb,
#                                               # vis_util.make_point_cloud_o3d(aa_box_points, color=[0., 0., 1.]),
#                                               # bandu_util.create_arrow(aa_source_vector, [0., 0., 1.]),
#                                               # bandu_util.create_arrow(oriented_source_vector, [0., 1., 0.]),
#                                               # bandu_util.create_arrow(R.from_matrix(rot_to_apply).apply(oriented_source_vector), [1., 0., 0.]),
#                                               # obb.rotate(rot_to_apply, np.zeros(3)),
#                                               vis_util.make_point_cloud_o3d(rotated_pointcloud, color=np.zeros(3)),
#                                               open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
#     return rot_to_apply
#
# from supervised_training.utils.visualization_util import make_colors
#
def get_relative_rotation_from_hot_labels(rotated_pointcloud,
                                             hot_labels,
                                             dir="surface_to_upright",
                                          min_z=None,
                                          max_z=None,
                                          ret_idxs=False):
    """

    :param rotated_pointcloud: num_points x 3
    :param hot_labels: num_points
    :param dir: whether we find the rotation from the surface normal to the -z, or the rotation from -z to the surface normal
    :param com: center of mass, used to determine oriented normal
    :param min_z: for debugging purposes
    :param max_z: for debugging purposes
    :return:
    """
    assert len(rotated_pointcloud.shape) == 2
    assert len(hot_labels.shape) == 1, hot_labels.shape
    assert dir in ["surface_to_upright", "upright_to_surface"]

    # since the hot labels == 1 means we have a BACKGROUND point, to get the surface points we must invert it

    surface_points = rotated_pointcloud[(~(hot_labels.bool())).byte()]

    # mc1 = make_colors(hot_labels)
    # geoms_to_draw = [
    #     vis_util.make_point_cloud_o3d(rotated_pointcloud,
    #                                   color=mc1),
    #     # vis_util.make_point_cloud_o3d(canonical_pc + np.array([0.,0.,.5]),
    #     #                               color=mc2),
    #     bandu_util.create_sphere(np.array([0., 0., min_z]), np.array([1., 0., 0.]), radius=.003),
    #     bandu_util.create_sphere(np.array([0., 0., max_z]), np.array([0., 1., 0.]), radius=.003),
    #     open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])]
    #
    #     # geoms_to_draw.append(bandu_util.create_arrow(plane_model[:3], [1., 0., 0.]))
    #     # geoms_to_draw.append(gen_surface_box(plane_model))
    #
    # open3d.visualization.draw_geometries(geoms_to_draw)


    # print("ln60 surface_points shape")
    # print(surface_points.shape)
    surface_pcd = open3d.geometry.PointCloud()
    surface_pcd.points = open3d.utility.Vector3dVector(surface_points.cpu().data.numpy())
    try:
        # print("ln68 num points")
        # print(surface_points.cpu().data.numpy().shape)
        plane_model, plane_idxs = surface_pcd.segment_plane(.007, 15, 1000)

        # print("ln162")
        # plane_model, plane_idxs = surface_pcd.segment_plane(.01, 15, 1000)
        plane_normal = np.array(plane_model)[:3]
        a, b, c, d = plane_model

        # orient the plane normal away from the center of mass
        # normal should have the same sign as the vector from the center of mass to the plane origin
        oriented_normal = np.sign(-d) * plane_normal

        # print("ln169")
        if dir == "surface_to_upright":
            if ret_idxs:
                return get_rotation_matrix_between_vecs([0, 0, -1], oriented_normal), plane_model, plane_idxs
            else:
                return get_rotation_matrix_between_vecs([0, 0, -1], oriented_normal), plane_model
        else:
            if ret_idxs:
                return get_rotation_matrix_between_vecs(oriented_normal, [0, 0, -1]), plane_model, plane_idxs
            else:
                return get_rotation_matrix_between_vecs(oriented_normal, [0, 0, -1]), plane_model
    except:
        # print("Unable to find any / enough points. This should never happen for one hot labels")
        raise Exception
#
#
# def gen_surface_box(plane_model, ret_centroid=False):
#     # rotate plane according to normal
#     rot_mat = bandu_util.get_rotation_matrix_between_vecs(plane_model[:3], [0, 0, 1])
#
#     box = open3d.geometry.TriangleMesh.create_box(width=.5, height=.5, depth=.003)
#
#     box.paint_uniform_color([1, 0.706, 0])
#
#
#     box.compute_vertex_normals()
#
#     # center box at COM
#     box.translate(-box.get_center())
#
#     box.rotate(rot_mat, center=np.zeros(3))
#
#     # surface_points = batch['rotated_pointcloud'][0][0][torch.sigmoid(predictions[0]) < .5]
#     # surface_pcd = open3d.geometry.PointCloud()
#     # surface_pcd.points = open3d.utility.Vector3dVector(surface_points.cpu().data.numpy())
#     # plane_model, plane_idxs = surface_pcd.segment_plane(.007, 15, 1000)
#     plane_normal = np.array(plane_model)[:3]
#
#     a, b, c, d = plane_model
#     origin_translation_vector = -d / (plane_normal @ plane_normal) * plane_normal
#
#     # translate it by the origin
#     box.translate(origin_translation_vector)
#
#     if ret_centroid:
#         return box, origin_translation_vector
#     else:
#         return box