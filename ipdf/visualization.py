"""Visualization functions. Adapted from """
from matplotlib import pyplot as plt
import numpy as np
from pytorch3d import transforms
import torch

def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                rotations_pred=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                show_color_wheel=True,
                                canonical_rotation=torch.eye(3)):
  """Plot a single distribution on SO(3) using the tilt-colored method.

  Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    rotations_pred: [3, 3] predicted rotation matrices.
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    display_threshold_probability: The probability threshold below which to omit
      the marker
    show_color_wheel: If True, display the explanatory color wheel which matches
      color on the plot with tilt angle

  Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
  """
  def _show_single_marker(ax, rotation, marker, edgecolors=True,
                          facecolors=False):
    eulers = transforms.matrix_to_euler_angles(rotation,
                                               convention='XYZ')
    xyz = rotation[:, 0]
    tilt_angle = eulers[0]
    longitude = np.arctan2(xyz[0], -xyz[1])
    latitude = np.arcsin(xyz[2])

    color = cmap(0.5 + float(tilt_angle) / 2 / np.pi)
    ax.scatter(longitude, latitude, s=2500,
               edgecolors=color if edgecolors else 'none',
               facecolors=facecolors if facecolors else 'none',
               marker=marker,
               linewidth=4)

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
  if rotations_gt is not None and len(rotations_gt.shape) == 2:
    rotations_gt = rotations_gt[None]

  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  eulers_queries = transforms.matrix_to_euler_angles(rotations,
                                                     convention='XYZ')
  xyz = rotations[:, :, 0]
  tilt_angles = eulers_queries[:, 0]

  longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
  latitudes = np.arcsin(xyz[:, 2])

  which_to_display = (probabilities > display_threshold_probability)

  if rotations_gt is not None:
    # The visualization is more comprehensible if the GT
    # rotation markers are behind the output with white filling the interior.
    display_rotations_gt = rotations_gt @ canonical_rotation

    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o')
    # Cover up the centers with white markers
    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o', edgecolors=False,
                          facecolors='#ffffff')

  if rotations_pred is not None:
    display_rotations_pred = rotations_pred @ canonical_rotation
    _show_single_marker(ax, display_rotations_pred, '8')
    _show_single_marker(ax, display_rotations_pred, '8', edgecolors=False,
                        facecolors='#ffffff')


  # Display the distribution
  ax.scatter(
      longitudes[which_to_display],
      latitudes[which_to_display],
      s=scatterpoint_scaling * probabilities[which_to_display],
      c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

  ax.grid()
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  if show_color_wheel:
    # Add a color wheel showing the tilt angle to color conversion.
    ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
    theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
    radii = np.linspace(0.4, 0.5, 2)
    _, theta_grid = np.meshgrid(radii, theta)
    colormap_val = 0.5 + theta_grid / np.pi / 2.
    ax.pcolormesh(theta, radii, colormap_val.T,
                  cmap=cmap,
                  shading='auto')
    ax.set_yticklabels([])
    ax.set_xticklabels([r'90$\degree$', None,
                        r'180$\degree$', None,
                        r'270$\degree$', None,
                        r'0$\degree$'], fontsize=14)
    ax.spines['polar'].set_visible(False)
    plt.text(0.5, 0.5, 'Tilt', fontsize=14,
             horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)

  return fig
