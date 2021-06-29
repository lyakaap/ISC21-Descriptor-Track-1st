# FB-ISC


## exp
- v1: simsiam
- v2: simsiam (aug片方だけ)
- v3: arcface
- v4: だんだんsample sizeを増やしていく
- v5: triplet loss
- v6: with miner
- v7: contrastive loss
- v8: Augly

python v1.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr --dim 256 --pred-dim 128 \
  --epochs 20 \
  --batch-size 128 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  /mnt/sdb/yokoo/fbisc/input/training_images

CUDA_VISIBLE_DEVICES=1 python v2.py \
  -a dm_nfnet_f0 \
  --batch-size 256 \
  --mode extract \
  --fix-pred-lr --dim 256 --pred-dim 128 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --extract-from proj \
  --weight ./v2/train/checkpoint_0001.pth.tar \
  /mnt/sdb/yokoo/fbisc/input/
CUDA_VISIBLE_DEVICES=2 python v3.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --weight ./v3/train/checkpoint_0019.pth.tar \
  /mnt/sdb/yokoo/fbisc/input/

python v2.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr --dim 256 --pred-dim 128 \
  --epochs 5 \
  --batch-size 128 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  /mnt/sdb/yokoo/fbisc/input/training_images

python v3.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 10 \
  --lr 0.01 --batch-size 64 --wd 1e-5 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  -s 30 -m 0.1 \
  --input-size 320 --sample-size 1000000 \
  /mnt/sdb/yokoo/fbisc/input/training_images

python v5.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 -m 0.05 \
  --input-size 320 \
  ../train_subset
CUDA_VISIBLE_DEVICES=2 python v5.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --weight ./v5/train/checkpoint_0000.pth.tar \
  --input-size 320 \
  --eval-subset \
  /mnt/sdb/yokoo/fbisc/input/

python v6.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 -m 0.1 --type-of-triplets semihard\
  --input-size 320 \
  ../train_subset
CUDA_VISIBLE_DEVICES=2 python v6.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --weight ./v6/train/checkpoint_0000.pth.tar \
  --input-size 320 \
  --eval-subset \
  /mnt/sdb/yokoo/fbisc/input/

python v7.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 2 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.2 --neg-margin 0.7 \
  --input-size 320 --sample-size 1000000 \
  /mnt/sdb/yokoo/fbisc/input/training_images
CUDA_VISIBLE_DEVICES=2 python v7.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v7/train/checkpoint_0001.pth.tar \
  --input-size 384 \
  /mnt/sdb/yokoo/fbisc/input/
python v8.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.03 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.0 --neg-margin 0.7 \
  --input-size 320 --sample-size 100000 \
  ../train_subset
CUDA_VISIBLE_DEVICES=2 python v8.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --weight ./v8/train/checkpoint_0000.pth.tar \
  --input-size 320 \
  --eval-subset \
  /mnt/sdb/yokoo/fbisc/input/

python ../scripts/eval_metrics.py v2/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv
python ../scripts/eval_metrics.py v5/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv

384x384:
{
  "average_precision": 0.15055434775303866,
  "recall_p90": 0.06952514526147065
}

{
  "average_precision": 0.23868405518343083,
  "recall_p90": 0.11099979963935083
}
python v5.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 -m 0.1 \
  --input-size 320 \
  ../train_subset

{
  "average_precision": 0.254530420701898,
  "recall_p90": 0.1825285513925065
}
python v7.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.2 --neg-margin 0.7 \
  --input-size 320 \
  ../train_subset

{
  "average_precision": 0.32818904246455216,
  "recall_p90": 0.2296133039471048
}
python v8.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.0 --neg-margin 0.7 \
  --input-size 320 --sample-size 100000 \
  ../train_subset
