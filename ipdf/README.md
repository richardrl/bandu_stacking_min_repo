# Bandu + IPDF integration

## Training from scratch

Saves eval figures and checkpoints to `/tmp` by default.

```sh
json_path="bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json"
train_path="bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"
val_path="bandu_project_data/val/fps_randomizenoiseTrue_numfps10_samples"
basedir=bandu_stacking_min_repo/supervised_training/

python $basedir/train_relativerotation.py --randomize_z_canonical --batch_size_train=24 --batch_size_eval 8 --dont_make_btb --lr=0.0001 --stats_json=$json_path $train_path $val_path --num_queries_train 2 --num_queries_eval 4 --query_sampling_mode grid --max_train_samples_per_epoch 1000000 --max_val_samples 1000000 --num_layers 4 --run_id l2lr1em4
```

## Resume training from a checkpoint

```sh
json_path="bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json"
train_path="bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"
val_path="bandu_project_data/val/fps_randomizenoiseTrue_numfps10_samples"
basedir=bandu_stacking_min_repo/supervised_training/

python $basedir/train_relativerotation.py --randomize_z_canonical --batch_size_train=24 --batch_size_eval 8 --dont_make_btb --lr=0.0001 --stats_json=$json_path $train_path $val_path --num_queries_train 2 --num_queries_eval 4 --query_sampling_mode grid --max_train_samples_per_epoch 1000000 --max_val_samples 1000000 --num_layers 4 --run_id l2lr1em4 --resume_pkl /tmp/bandu_ipdf_ckpt_l2lr1em4_e40 --first_epoch 40
```

## Running pre-trained model on some input point cloud.

```python
import torch

from ipdf import models as ipdf_models
from ipdf import visualization
from supervised_training.models.dgcnn_cls import DGCNNCls

checkpoint = 'bandu_ipdf_ckpt_l2lr1em4_e40'

# Load models.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DGCNNCls(num_class=256).to(device)
ipdf = ipdf_models.ImplicitPDF(feature_size=256, num_layers=2).to(device)

pkl = torch.load(checkpoint, map_location=device)
model.load_state_dict(pkl['model'])
ipdf.load_state_dict(pkl['ipdf'])

model.eval()
ipdf.eval()

# Make dummy point cloud input.
point_cloud = torch.rand(1, 1, 2048, 3)

# Apply models.
with torch.no_grad():
  feature = model(point_cloud.squeeze(1).permute(0, 2, 1))
  # Recursion level 4 amounts to ~300k samples.
  queries = ipdf_models.generate_queries(
    num_queries=4,
    mode='grid',
    rotate_to=torch.eye(3, device=device)[None])
  pdf, pmf = ipdf.compute_pdf(feature, queries)

  # If we have to output a single rotation, this is it.
  # TODO: we could run gradient ascent here to improve accuracy. 
  predicted_pose = queries[0][pdf.argmax(axis=-1)]

  # Show probability map. For a random point cloud this should be
  # quite dense, indicating high uncertainty.
  fig = visualization.visualize_so3_probabilities(
    queries[0].cpu(),
    pmf[0].cpu(),
    rotations_pred=predicted_pose[0].cpu())
  fig.show()
```
