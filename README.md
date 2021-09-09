(Training working based off the below command)

## Installation
Create new virtual env, and install from requirements.txt.

## Syntax used in comments to describe dimensions

```$xslt
nB: batch dimension
nO: object dimension (always 1, for now. Will squeeze away this dimension often)
num_points: number of points in pointcloud
```

## Training
1. Example (replace the last two paths with the paths to your training and test samples folder):
```$xslt

python supervised_training/train_relativerotation.py --stats_json="/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json" "/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"  "/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"
```

2. Changing or utilizing the following lines should be necessary for implementing implicit PDF:

In supervised_training/train_relativerotation.py:
- model = DGCNNCls(num_class=4)
- predictions = model(batch['rotated_pointcloud'].squeeze(1))

## Dataset
The dataset class file is here: supervised_training/dataset.py

You can initialize each dataset by feeding in the path to the samples directory, e.g. 
"/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"

## Dataset Batch
Each batch in the dataset is a dictionary with multiple values.

```
batch['rotated_pointcloud']: 
- These are the pointclouds in the starting orientation, on the table.
batch['canonical_pointcloud']: 
- These are the pointclouds that are in the canonical/identity/stacked orientation. 
The reason why the identity orientation pointclouds are stacked, is because each object's 
identity orientation when loading into pyBullet is the stacked orientation. 
After dropping the object randomly on the table, we log a new starting orientation in PyBullet.

batch['rotated_quat']: 
- These are the orientations of the above 'rotated_pointcloud' in PyBullet.
Since we want to apply a world-frame rotation to this rotated_pointcloud, such that it becomes
the canonical_pointcloud (and this rotation corresponds to the object rotating from the starting orientation to the stacked orientation),
the regression target for our neural network is the *inverse* of this 'rotated_quat':
-- R.from_quat(batch['rotated_quat']).inv()
```

## Visualizing samples from dataset
