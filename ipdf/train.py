"""Main training loop."""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torchvision import models as torchvision_models

from ipdf import datasets
from ipdf import models
from ipdf import visualization


def parse_args():
  parser = argparse.ArgumentParser(description='Implicit PDF on pytorch')

  parser.add_argument("--dataset", default='dummy',
                      choices=['dummy', 'symsol/cube', 'symsol1'])
  parser.add_argument("--num_steps", type=int, default=10_000,
                      help="number of steps to train.")
  parser.add_argument("--num_eval_steps", type=int, default=100,
                      help="number of steps to eval.")

  parser.add_argument("--eval_every", type=int, default=1000,
                      help="eval every # steps.")

  parser.add_argument("--batch_size_train", type=int, default=32)
  parser.add_argument("--batch_size_eval", type=int, default=32)
  parser.add_argument("--learning_rate", type=float, default=1e-4)

  parser.add_argument("--num_queries_train", type=int, default=256,
                      help="number of random queries per input during training, or recursion level for grid.")
  # Eval is always on a grid; recursion level 2 corresponds to ~4k points.
  parser.add_argument("--num_queries_eval", type=int, default=2,
                      help="see `num_queries_train`.")
  parser.add_argument("--query_sampling_mode", default='random',
                      choices=['random', 'grid'],
                      help="how to sample rotation queries.")
  parser.add_argument("--num_layers", type=int, default=2,
                      help="Number of IPDF MLP layers.")

  return parser.parse_args()


def batch_to_torch(batch, device):
  if isinstance(batch['pose'], tf.Tensor):
    batch = {k: torch.from_numpy(v.numpy())
             for k, v in batch.items()}
  # Send data to device.
  return {k: (v.to(device))
          for k, v in batch.items()}


def get_input_features(backbone, batch):
  # If feature is already provided, as in the dummy dataset, use
  # it. Else apply backbone to inputs.
  if batch.get('feature') is not None:
    return batch['feature']
  return backbone(batch['input'])


def find_closest_rotations(a, b):
  """Return the indices of rotations in `a` that are closest to `b`.

  Args:
    a: (num_a, 3, 3) rotation matrices.
    b: (num_b, 3, 3) rotation matrices.
  Returns:
    c: (num_b,) indices where a[c[i]] is the closest rotation to b[i].
  """
  # Computes A^T B
  relative_rotation = torch.einsum('aji,bjk->baik', a, b)
  traces = torch.einsum('abii', relative_rotation)
  # Max trace is closest rotation.
  return traces.argmax(axis=-1)


def train_step(args, model, backbone, batch, optimizer):
  optimizer.zero_grad()
  model.train()
  backbone.train()
  feature = get_input_features(backbone, batch)
  queries = models.generate_queries(num_queries=args.num_queries_train,
                                    mode=args.query_sampling_mode,
                                    rotate_to=batch['pose'])
  pdf, pmf = model.compute_pdf(feature, queries)
  # Loss is negative log-likelihood of the ground truth, which is
  # the last query.
  loss = -torch.log(pdf[:, -1]).mean()
  loss.backward()
  optimizer.step()

  return loss, queries, pmf


def save_figure(path, queries, pmf, gt, pred=None):
  fig = visualization.visualize_so3_probabilities(
    queries,
    pmf,
    rotations_gt=gt,
    rotations_pred=pred,
    display_threshold_probability=1/queries.shape[0])
  fig.savefig(path)
  plt.close(fig)


def compute_angle_error(R1, R2):
    """Angle error between batch of (batch_size, 3, 3) rotation matrices."""
    R = R1.transpose(1, 2) @ R2
    trace = torch.einsum('aii', R)
    return torch.arccos(torch.clip((trace - 1)/2, -1.0, 1.0))


def eval_step(args, model, backbone, dataset, device, step):
  model.eval()
  backbone.eval()
  with torch.no_grad():
    log_likelihoods = []
    angle_errors = []
    for e_step, batch in enumerate(dataset):
      batch = batch_to_torch(batch, device)
      feature = get_input_features(backbone, batch)
      # During evaluation we do not rotate the queries, and always use
      # 'grid' mode.
      queries = models.generate_queries(
        num_queries=args.num_queries_eval,
        mode='grid',
        rotate_to=torch.eye(3, device=device)[None])
      pdf, pmf = model.compute_pdf(feature, queries)
      # Take likelihood of closest point to ground truth.
      closest = find_closest_rotations(queries[0], batch['pose'])
      pdf_closest = pdf[torch.arange(pdf.shape[0]), closest]
      log_likelihoods.append(torch.log(pdf_closest).mean().item())

      # If we have to output a single rotation prediction,
      # this is it.
      mode = queries[0][pdf.argmax(axis=-1)]
      # Compute distance between argmax pdf and gt pose.
      # This is innacurate in case of symmetric objects!
      angle_errors.append(
        compute_angle_error(mode, batch['pose']).mean().item())

      if e_step + 1 >= args.num_eval_steps:
        break

    # Save figures for elements of the last batch.
    for i in range(3):
      save_figure(path=f'/tmp/eval_ipdf_{i}_{step:04d}.png',
                  queries=queries[0].cpu(),
                  pmf=pmf[i].cpu(),
                  gt=batch['pose'][i].cpu(),
                  pred=mode[i].cpu())

  return np.mean(log_likelihoods), np.mean(angle_errors)


def train_and_evaluate(args):
  start = time.time()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # TODO: take as param?
  feature_size = 256
  backbone = torchvision_models.resnet18(pretrained=True)
  backbone.fc = nn.Linear(backbone.fc.in_features, feature_size)
  model = models.ImplicitPDF(feature_size=feature_size,
                             num_layers=args.num_layers)

  backbone = backbone.to(device)
  model = model.to(device)

  optimizer = torch.optim.Adam(list(model.parameters()) +
                               list(backbone.parameters()),
                               lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         args.num_steps)
  batch_size = {'train': args.batch_size_train,
                'eval': args.batch_size_eval}
  dataset = datasets.load_dataset(args.dataset,
                                  batch_size,
                                  feature_size=feature_size)
  for step, batch in enumerate(dataset['train']):
    if step >= args.num_steps:
      break
    batch = batch_to_torch(batch, device)
    loss, queries, pmf = train_step(args, model, backbone, batch, optimizer)
    scheduler.step()

    if step % 100 == 0:
      # Report training results.
      elapsed = time.time() - start
      lr = scheduler.get_lr()[0] * 1e4
      print(f'Elapsed={elapsed:.1f}s, step {step}, train loss: {loss.item():.2f}, lr={lr:.2f}e-4')
      # Save examples from last training batch.
      with torch.no_grad():
        for i in range(3):
          save_figure(path=f'/tmp/train_ipdf_{i}_{step:04d}.png',
                      queries=queries[i].cpu(),
                      pmf=pmf[i].cpu(),
                      gt=batch['pose'][i].cpu())
    # TODO: use step+1 to avoid evaluating the initialization
    if step % args.eval_every == 0:
      # Evaluate.
      gt_log_likelihood, angle_errors = eval_step(
        args, model, backbone, dataset['eval'], device, step)
      elapsed = time.time() - start
      print(f'Elapsed={elapsed:.1f}s, step {step}, '
            f'eval log-likelihood: {gt_log_likelihood:.2f} '
            f'eval angle error: {np.degrees(angle_errors):.2f} deg.')


if __name__ == '__main__':
  args = parse_args()
  train_and_evaluate(args)
