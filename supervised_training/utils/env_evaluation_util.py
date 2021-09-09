import numpy
import torch
from supervised_training.utils import surface_util
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d

def get_bti(batched_pointcloud,
            threshold_bottom_of_upper_region,
            threshold_top_of_bottom_region=None,
            max_z=None, min_z=None):
    """

    :param batched_pointcloud:
    :param threshold_bottom_of_upper_region: Bottom of upper region, expressed as fraction of total object height
    :param threshold_top_of_bottom_region: Top of bottom region, expressed as fraction of total object height
    :param line_search: search along the canonical axis for the section with the most support
    :return:
    """

    if threshold_top_of_bottom_region is not None:
        assert threshold_top_of_bottom_region < threshold_bottom_of_upper_region

    # returns boolean where 1s rotated_pc_mean BACKGROUND and 0 means surface

    # since pointcloud is in the canonical position, we chop off the bottom points
    if max_z is None and min_z is None:
        max_z = np.max(batched_pointcloud[..., -1], axis=-1)
        min_z = np.min(batched_pointcloud[..., -1], axis=-1)
        object_heights = (max_z - min_z)
    else:
        object_heights = max_z - min_z

    threshold_object_height = object_heights * threshold_bottom_of_upper_region
    threshold_world_bottom_of_upper_region = min_z + threshold_object_height
    bti = np.greater(batched_pointcloud[..., -1], np.expand_dims(threshold_world_bottom_of_upper_region, axis=-1))

    if threshold_top_of_bottom_region:
        threshold_world_top_of_bottom_region = min_z + object_heights * threshold_top_of_bottom_region
        bti_bottom_region = np.less(batched_pointcloud[..., -1], np.expand_dims(threshold_world_top_of_bottom_region, axis=-1))

        bti = bti + bti_bottom_region
    bti = bti[..., None]

    return bti


def get_bti_from_rotated(rotated_batched_pointcloud, orientation_quat, threshold_frac,
                         linear_search=False,
                         max_z=None, min_z=None,
                         max_frac_threshold=.1
                         ):
    """
    Gets a single bti
    :param rotated_batched_pointcloud:
    :param orientation_quat:
    :param threshold_frac: Fraction of height to use the threshold on
    :return:
    """
    assert len(rotated_batched_pointcloud.shape) == 2
    assert len(orientation_quat.shape) == 1

    # canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud.cpu().data.numpy())
    canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud)

    if linear_search:
        # do a linear search over 100 pairs for the section that gives best oriented normal, for the CANONICAL pointcloud,
        # in terms of cosine distance to the negative gravity vector
        # delineate [lower, upper] values from 0% to 10%

        # keep increasing the threshold until you get enough points

        def find_bti(threshold_frac_inner):
            found_btis = []
            found_rotmats_distance_to_identity = []

            for lower_start_frac in np.linspace(0, max_frac_threshold, num=100):
                found_bti = get_bti(canonical, threshold_frac_inner + lower_start_frac, lower_start_frac, max_z=max_z, min_z=min_z)

                try:
                    relative_rotmat, plane_model = surface_util.get_relative_rotation_from_hot_labels(torch.as_tensor(canonical),
                                                                                                      torch.as_tensor(found_bti.squeeze(-1)),
                                                                                                      min_z=min_z,
                                                                                                      max_z=max_z)
                except:
                    continue
                # print(relative_rotmat)
                found_rotmats_distance_to_identity.append(np.linalg.norm(relative_rotmat - np.eye(3)))
                found_btis.append(found_bti)
            return found_rotmats_distance_to_identity, found_btis

        try:
            for threshold_frac_inner in [threshold_frac, 2*threshold_frac, 4*threshold_frac, 5*threshold_frac]:
                outer_found_rotmats_distance_to_identity, outer_found_btis = find_bti(threshold_frac_inner)

                if outer_found_rotmats_distance_to_identity:
                    closest_to_identity_id = np.argmin(outer_found_rotmats_distance_to_identity)
                    break

            assert np.sum(outer_found_btis[closest_to_identity_id]) > 0, np.sum(outer_found_btis[closest_to_identity_id])
            assert np.sum(outer_found_btis[closest_to_identity_id]) < outer_found_btis[closest_to_identity_id].shape[0], \
                np.sum(outer_found_btis[closest_to_identity_id])
            return outer_found_btis[closest_to_identity_id]
        except:
            return get_bti(canonical, threshold_frac, 0, min_z=min_z,
                           max_z=max_z)
    else:
        return get_bti(canonical, threshold_frac, 0, min_z=min_z,
                       max_z=max_z)