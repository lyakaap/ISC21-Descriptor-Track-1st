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
- v23: v19mのつづき、res=448x448, lr-0.01から、aug少し激しく
- v24: neg_margin=1.0
- v25: swin
- v26: data aug 強め
- v27: eff-l, tuned, 10epoch
- v28: schedulerをiterごとにstep
- v29: RAっぽく
- v30: + RandomRotation
- v31: v30 + linear, eff-l
- v32: v31 + overlay image
- v33: v31 + swin
- v34: v30 + linear, eff-m
- v35: v30 + linear, eff-s, bs=64
- v36: v31のつづき、res=, lr-0.01から、aug少し激しく

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
  --lr 0.01 --wd 1e-5 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.1 \
  --input-size 448 --sample-size 1000000 --memory-size 10000 \
  --weight ./v19m/train/checkpoint_0004.pth.tar \
  ../input/training_images/
for epoch in `seq 0 4`; do
  python v23.py -a tf_efficientnetv2_m_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v23/train/checkpoint_000${epoch}.pth.tar --input-size 448 --eval-subset ../input/
done
python v23.py \
  -a tf_efficientnetv2_m_in21k \
  --batch-size 256 \
  --mode extract --target-set qrt \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v23/train/checkpoint_0004.pth.tar \
  --input-size 448 \
  ../input/

python v27.py \
  -a tf_efficientnetv2_s_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.35 --wd 1e-6 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
for epoch in `seq 0 4`; do
  python v27.py -a tf_efficientnetv2_s_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v27/train/checkpoint_000${epoch}.pth.tar --input-size 256 --eval-subset ../input/
done

python v31.py \
  -a tf_efficientnetv2_l_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.1 --wd 1e-6 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
python v31.py \
  -a tf_efficientnetv2_l_in21k \
  --batch-size 256 \
  --mode extract --target-set qr \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v31/train/checkpoint_0004.pth.tar \
  --input-size 256 \
  ../input/

python v33.py \
  -a swin_base_patch4_window7_224_in22k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.0003 --wd 1e-2 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 10000 \
  ../train_subset/
for epoch in `seq 0 4`; do
  python v33.py -a swin_base_patch4_window7_224_in22k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v33/train/checkpoint_000${epoch}.pth.tar --input-size 224 --eval-subset ../input/
done

python v34.py \
  -a tf_efficientnetv2_m_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.1 --wd 1e-6 \
  --batch-size 128 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
gsutil -m cp -r v34 gs://fbisc/exp/
sudo shutdown

for epoch in `seq 0 4`; do
  python v34.py -a tf_efficientnetv2_m_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v34/train/checkpoint_000${epoch}.pth.tar --input-size 256 --eval-subset ../input/
done
python v35.py \
  -a tf_efficientnetv2_s_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 \
  --lr 0.1 --wd 1e-6 \
  --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
for epoch in `seq 0 4`; do
  python v35.py -a tf_efficientnetv2_s_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v35/train/checkpoint_000${epoch}.pth.tar --input-size 256 --eval-subset ../input/
done

python v36.py \
  -a tf_efficientnetv2_l_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 \
  --lr 0.01 --wd 1e-6 \
  --batch-size 96 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 10000 \
  --weight ./v31/train/checkpoint_0004.pth.tar \
  ../input/training_images/
gsutil -m cp -r v36 gs://fbisc/exp/
sudo shutdown

for epoch in `seq 0 4`; do
  python v36.py -a tf_efficientnetv2_l_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v36/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done
python v36.py \
  -a tf_efficientnetv2_l_in21k \
  --batch-size 256 \
  --mode extract --target-set qrt \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v36/train/checkpoint_0004.pth.tar \
  --input-size 384 \
  ../input/

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
v23
{
  "average_precision": 0.4780176024970843,
  "recall_p90": 0.3129633340012022
}
v23 w/ embedding isolation
{
  "average_precision": 0.5565125454217211,
  "recall_p90": 0.46924464035263475
}
v23 w/ norm
{
  "average_precision": 0.5564276273733336,
  "recall_p90": 0.4738529352835103
}
v31
{
  "average_precision": 0.49870458464635814,
  "recall_p90": 0.3818873973151673
}
v31, eff-l
{
  "average_precision": 0.5095199804428568,
  "recall_p90": 0.42276096974554195
}
---------------------------------------------------------------



## ref
https://github.com/TengdaHan/ShuffleBN
https://github.com/facebookresearch/simsiam
