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

python $basedir/train_relativerotation.py --randomize_z_canonical --batch_size_train=24 --batch_size_eval 8 --dont_make_btb --lr=0.0001 --stats_json=$json_path $train_path $val_path --num_queries_train 2 --num_queries_eval 4 --query_sampling_mode grid --max_train_samples_per_epoch 1000000 --max_val_samples 1000000 --num_layers 4 --run_id l2lr1em4 --resume_pkl /tmp/bandu_ipdf_ckpt_l2lr1em4_e40
```

## Running pre-trained model on some point cloud

```sh
TODO
```
