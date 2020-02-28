# Evaluation method

要完成evaluation，先把你要evaluate的pth训练结果文件拷贝（不要移动）到test文件夹，evaluate完成后可以删除

## This is an example on how to run the evaluation on UCF101 (default dataset)
```
python evaluate_video.py \
--task-name ../exps/models/archive/0704_6-baseline/AR_PyTorch \
--log-file ./eval-ucf101-split1-190706.log --load-epoch 70
