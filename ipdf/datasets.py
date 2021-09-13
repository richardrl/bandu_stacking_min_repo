from torch.utils.data import Dataset
import numpy as np
from pytorch3d import transforms
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader


class DummyDataset(Dataset):
  """ Dummy dataset: feature is constant and pose changes according to
      a simple distribution.
  """
  def __init__(self, feature, poses, size):
    self.feature = feature
    self.poses = poses
    self.probabilities = [0.1, 0.3, 0.6]
    self.size = size

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    pose_id = np.random.choice(np.arange(len(self.poses)),
                               p=self.probabilities)
    return {'feature': self.feature,
            'pose': self.poses[pose_id]}


def load_symsol(shape_indices, split, batch_size):
  """Loads the symmetric_solids dataset.

  Adapted from
  https://github.com/google-research/google-research/blob/master/implicit_pdf/data.py.

  Args:
    split: 'train' or 'test', determining the split of the dataset.
    shape_indices: filter symsol by these indices.
    batch_size: batch size.

  Returns:
    tf.data.Dataset of images with the associated rotation matrices.
  """
  dataset = tfds.load('symmetric_solids', split=split)

  # Filter the dataset by shape index, and use the full set of equivalent
  # rotations only if split == test
  dataset = dataset.filter(
      lambda x: tf.reduce_any(tf.equal(x['label_shape'], shape_indices)))

  # We do not evaluate against all equivalent rotations here.
  # annotation_key = 'rotation' if split == 'train' else 'rotations_equivalent'

  def parse(example):
    image = tf.image.convert_image_dtype(example['image'], tf.float32)
    # torchvision resnet wants channels as the second dimension
    image = tf.transpose(image, (2, 0, 1))
    output = {'input': image, 'pose': example['rotation']}
    # if example.get('rotations_equivalent') is not None:
    #   output['pose_equivalent'] = example['rotations_equivalent']
    return output

  dataset = dataset.map(parse,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if split == 'train':
    # Things run much faster without shuffle! Dataset is shuffled once
    # but I think removing this extra shuffle here leads to
    # dataset = dataset.repeat().shuffle(1000).batch(batch_size)
    dataset = dataset.repeat().batch(batch_size)
  elif split == 'test':
    dataset = dataset.batch(batch_size)
  else:
    raise ValueError(f'Unknown split: {split}')

  return dataset


def load_dataset(name, batch_size, feature_size=None):
  if name == 'dummy':
    dataset = {}
    feature = np.random.rand(feature_size).astype(np.float32)
    poses = transforms.random_rotations(3)
    for phase, bs in batch_size.items():
      dataset[phase] = DataLoader(
          DummyDataset(feature=feature,
                       poses=poses,
                       size=100_000 if phase == 'train' else 100),
          batch_size=bs,
          shuffle=True if phase == 'train' else False)
  elif name.startswith('symsol'):
    if name == 'symsol/cube':
      shape_indices = [1]
    elif name == 'symsol1':
      shape_indices = np.arange(5)
    dataset = {}
    for phase, bs in batch_size.items():
      split = phase
      if phase == 'eval':
        split = 'test'
      dataset[phase] = load_symsol(shape_indices=shape_indices,
                                   split=split,
                                   batch_size=bs)
  else:
    raise ValueError(f'Unknown dataset: {name}')

  return dataset
