"""IPDF models."""
import functools
import healpy as hp
import numpy as np
import torch
from torch import nn
from pytorch3d import transforms


# This is adapted from
# https://github.com/google-research/google-research/blob/master/implicit_pdf/models.py
@functools.lru_cache()
def generate_healpix_grid(recursion_level, device='cpu'):
  """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

  Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
  along the 'tilt' direction 6*2**recursion_level times over 2pi.

  Args:
    recursion_level: An integer which determines the level of resolution of the
      grid.  The final number of points will be 72*8**recursion_level.  A
      recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
      for evaluation.

  Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
  """
  assert recursion_level <= 6
  number_per_side = 2**recursion_level
  number_pix = hp.nside2npix(number_per_side)
  s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
  s2_points = np.stack([*s2_points], 1)

  # Take these points on the sphere and
  azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
  tilts = np.linspace(0, 2*np.pi, 6*2**recursion_level, endpoint=False)
  polars = np.arccos(s2_points[:, 2])
  grid_rots_mats = []
  for tilt in tilts:
    # Build up the rotations from Euler angles, zyz format
    rot_mats = transforms.euler_angles_to_matrix(
      torch.stack([torch.from_numpy(azimuths.astype(np.float32)),
                   torch.zeros(number_pix),
                   torch.zeros(number_pix)], 1),
      convention='ZYX')

    rot_mats = (rot_mats @
                transforms.euler_angles_to_matrix(
                  torch.stack([torch.zeros(number_pix),
                               torch.zeros(number_pix),
                               torch.from_numpy(polars.astype(np.float32))],
                              1),
                  convention='ZYX'))

    rot_mats = (rot_mats @
                transforms.euler_angles_to_matrix(torch.Tensor([tilt, 0., 0.]),
                                                  convention='ZYX')[None])
    grid_rots_mats.append(rot_mats)

  grid_rots_mats = torch.cat(grid_rots_mats, 0)
  return grid_rots_mats.to(device)


def generate_queries(num_queries, mode, rotate_to):
  """Generate pose queries for IPDF.

  Args:
    num_queries: int.
    mode: str, sampling mode.
    rotate_to: (batch_size_or_1, 3, 3) rotation matrices. The last
      element of each set of queries will be rotate to the corresponding
      pose in `rotate_to`.
  Returns:
    (batch_size, num_queries, 3, 3)
  """
  if mode == 'random':
    queries = transforms.random_rotations(num_queries)
  elif mode == 'grid':
    queries = generate_healpix_grid(num_queries, rotate_to.device)
  else:
    raise ValueError(f'Sampling mode {mode} not supported!')

  queries = queries.to(rotate_to.device)
  # Rotation that applied to last element results in `rotate_to`.
  rotation_to_apply = queries[-1].T @ rotate_to
  # queries is (num_queries, 3, 3)
  # rotation_to_apply is (batch_size, 3, 3)
  # Do pairwise matmul.
  return torch.einsum('qij,bjk->bqik', queries, rotation_to_apply)


class ImplicitPDF(nn.Module):
  def __init__(self, feature_size=128, num_channels=256, num_layers=2):
    super().__init__()
    self.feature_embedding = nn.Linear(feature_size, num_channels)
    # query input is a flattened 3x3 rotation matrix.
    self.query_embedding = nn.Linear(9, num_channels)
    self.combined_relu = nn.ReLU()
    layers = []
    for l in range(num_layers - 1):
      layers.append(nn.Linear(num_channels, num_channels))
      layers.append(nn.ReLU())
    # Final layer
    layers.append(nn.Linear(num_channels, 1))

    self.mlp = nn.Sequential(*layers)

  def compute_log_probabilities(self, features, queries):
    """Compute unnormalized log-probabilities.

    Args:
      features: (batch_size, feature_size) array of input features (for
        example, deep image / point cloud descriptors)
      queries: (batch_size_or_1, num_queries, 3, 3) array of rotation
        matrices. Batch size can be 1 if same the same queries are to
        be used for all batch elements (usually done during
        evaluation).
    Returns:
      A (batch_size, num_queries) array of unnormalized log-probabilities
    """
    assert features.ndim == 2
    assert queries.ndim == 4
    assert queries.shape[-2:] == (3, 3)
    assert queries.shape[0] in [1, features.shape[0]]
    features = self.feature_embedding(features)
    queries = torch.reshape(queries, queries.shape[:2] + (9,))
    queries = self.query_embedding(queries)
    combined = self.combined_relu(features[:, None] + queries)
    return self.mlp(combined)[..., 0]

  def compute_pmf(self, features, queries):
    """Compute probability mass function.

    Values correspond to a discrete ditribution where each cell is
    centered on one element of `queries`, so values sum to 1.

    See also: `compute_log_probabilities`, `compute_pdf`.
    """
    log_probabilities = self.compute_log_probabilities(features, queries)
    return nn.Softmax(dim=-1)(log_probabilities)

  def compute_pdf(self, features, queries):
    """Compute probability density function.

    Values assume a continuous ditribution sampled at queries,
    therefore the sum is not necessarily 1.

    See also: `compute_log_probabilities`, `compute_pmf`.
    """
    pmf = self.compute_pmf(features, queries)
    # Divide by cell area here.
    return pmf * queries.shape[1] / np.pi**2, pmf
