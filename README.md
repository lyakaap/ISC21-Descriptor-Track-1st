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
- v9: Augly fixed
- v10: XBM
- v11: Bipartite contrastive loss
- v12: XBM only enqueue noaug
- v13: Bipartite XBM
- v14: XBM + NTXent
- v15: XBM + SignalToNoiseRatioContrastiveLoss
- v16: v10 + w/o bn
- v17: v10 + deeper head
- v18: v10 + different backbone
- v19: v18 + amp
- v20: v19, no sync bn
- v21: v20 + BN shuffle
- v22: v19 + double aug hard
- v23: v19mのつづき、res=384x384, aug少し激しく


python v8.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 32 --ncrops 4 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
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
python v8.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 10 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 320 --sample-size 1000000 \
  ../input/training_images
python v8.py \
  -a dm_nfnet_f0 \
  --batch-size 512 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v8/train/checkpoint_0004.pth.tar \
  --input-size 320 \
  ../input/
python v10.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../train_subset
CUDA_VISIBLE_DEVICES=2 python v10.py \
  -a dm_nfnet_f0 \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v10/train/checkpoint_0000.pth.tar \
  --input-size 256 \
  --eval-subset \
  ../input/
python v18.py \
  -a tf_efficientnetv2_s_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 1 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../train_subset
CUDA_VISIBLE_DEVICES=2 python v18.py \
  -a tf_efficientnetv2_s_in21k \
  --batch-size 128 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v18/train/checkpoint_0000.pth.tar \
  --input-size 256 \
  --eval-subset \
  ../input/
python v19.py \
  -a tf_efficientnetv2_s_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v19.py \
  -a tf_efficientnetv2_s_in21k \
  --batch-size 512 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v19/train/checkpoint_0004.pth.tar \
  --input-size 256 \
  --eval-subset \
  ../input/
python v19.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.02 --wd 1e-5 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v19.py \
  -a tf_efficientnetv2_m_in21k \
  --batch-size 512 \
  --mode extract --target-set t \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v19m/train/checkpoint_0004.pth.tar \
  --input-size 256 \
  ../input/

python v20.py \
  -a tf_efficientnetv2_m_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 1 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../input/training_images/
python v20.py \
  -a tf_efficientnetv2_m_in21k \
  --batch-size 1024 \
  --mode extract \
  --gem-p 3.0 --gem-eval-p 6.0 \
  --weight ./v20/train/checkpoint_0000.pth.tar \
  --input-size 256 \
  --eval-subset \
  ../input/
python v19.py \
  -a tf_efficientnetv2_m_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
for epoch in `seq 0 4`; do
  python v19.py -a tf_efficientnetv2_m_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v19/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done

python v23.py \
  -a tf_efficientnetv2_m_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 384 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
for epoch in `seq 0 4`; do
  python v23.py -a tf_efficientnetv2_m_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v23/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done

python ../scripts/eval_metrics.py v2/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv

384x384:
{
  "average_precision": 0.15055434775303866,
  "recall_p90": 0.06952514526147065
}
v8, 320x320
{
  "average_precision": 0.2829796970097667,
  "recall_p90": 0.07313163694650371
}

v19, 256x256
{
  "average_precision": 0.4035323730537874,
  "recall_p90": 0.21598877980364656
}
v19m, 256x256
{
  "average_precision": 0.44403655362651634,
  "recall_p90": 0.2484472049689441
}
---------------------------------------------------------------

{
  "average_precision": 0.4242279033736717,
  "recall_p90": 0.29252654778601483
}
python v8.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 4.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 320 --sample-size 100000 \
  ../train_subset

{
  "average_precision": 0.4604716294783043,
  "recall_p90": 0.33099579242636745
}
python v10.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 320 --sample-size 100000 --memory-size 4096 \
  ../train_subset

{
  "average_precision": 0.45031799005000156,
  "recall_p90": 0.2785013023442196
}
python v10.py \
  -a dm_nfnet_f0 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --epochs 1 \
  --lr 0.01 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../train_subset

{
  "average_precision": 0.49208326754843634,
  "recall_p90": 0.3698657583650571
}
python v18.py \
  -a tf_efficientnetv2_s_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 1 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../train_subset

{
  "average_precision": 0.4996100416157455,
  "recall_p90": 0.3742736926467642
}
python v19.py \
  -a tf_efficientnetv2_m_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 1 \
  --lr 0.1 --wd 1e-5 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 256 --sample-size 100000 --memory-size 4096 \
  ../input/training_images/

eff
{
  "average_precision": 0.4574111312254072,
  "recall_p90": 0.32919254658385094
}
{
  "average_precision": 0.49031745636019775,
  "recall_p90": 0.3696653977158886
}
{
  "average_precision": 0.5014965338316312,
  "recall_p90": 0.38068523342015625
}
{
  "average_precision": 0.5051015215892052,
  "recall_p90": 0.3816870366659988
}
{
  "average_precision": 0.5059337166831631,
  "recall_p90": 0.38088559406932476
}

nfnet
{
  "average_precision": 0.4613350180144992,
  "recall_p90": 0.32478461230214384
}
{
  "average_precision": 0.45713352189285605,
  "recall_p90": 0.34662392306151074
}
{
  "average_precision": 0.48193417234902997,
  "recall_p90": 0.35584051292326185
}
{
  "average_precision": 0.49183142728298346,
  "recall_p90": 0.36545782408335004
}
{
  "average_precision": 0.4948734368558382,
  "recall_p90": 0.3696653977158886
}

## ref
https://github.com/TengdaHan/ShuffleBN
https://github.com/facebookresearch/simsiam
