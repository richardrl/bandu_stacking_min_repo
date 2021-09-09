(currently still modifying this, training not working yet)

## Installation
1. Create new virtual env, and install from requirements.txt.

## Training
1. Example:
```$xslt

python supervised_training/train_relativerotation.py --stats_json="/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json" "/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"  "/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"
```

2. Changing the following lines should be necessary for implementing implicit PDF:

In supervised_training/train_relativerotation.py:
- model = DGCNNCls(num_class=4)
- 

## Dataset
The dataset class file is here: supervised_training/dataset.py

You can initialize each dataset by feeding in the path to the samples directory, e.g. 
"/home/richard/Dropbox (MIT)/bandu_project_data/train/fps_randomizenoiseTrue_numfps10_samples"

## Visualizing samples from dataset
