# Evaluation method

## This is an example on how to run the evaluation on UCF101 (default dataset)
```
python evaluate_video.py \
--task-name ../exps/models/archive/0704_6-baseline/AR_PyTorch \
--log-file ./eval-ucf101-split1-190706.log --load-epoch 70
