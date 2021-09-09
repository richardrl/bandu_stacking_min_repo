(currently still modifying this, training not working yet)

## Installation
1. Create new virtual env, and install from requirements.txt.

## Syntax used in comments to describe dimensions

```$xslt
nB: batch dimension
nO: object dimension (always 1, for now. Will squeeze away this dimension often)
num_points: number of points in pointcloud
```

## Training
1. Example:
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

## Visualizing samples from dataset
