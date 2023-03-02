#!/bin/bash/bin/bash
python main.py --data_dir=../KITTI/raw/ --batch_size=4  --mode=depth --train_test=train  --retrain=True  --train_file=./filenames/kitti_train_files_png_4frames.txt --gt_2012_dir=../KITTI/2012/training/ --gt_2015_dir=../KITTI/2015/training/ --pretrained_model=./Model_Logs/checkpoint  --trace=./Model_Logs
