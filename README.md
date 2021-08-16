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
- v36: v31のつづき、res=, lr=0.01から、aug少し激しく
- v37: v36, lr=0.05から
- v38: v36のつづき
- v39: p=1.0
- v40: p=2.0
- v41: p=3.0
- v42: new subset baseline
- v43: Edge enhance
- v44: Edge enhance + instagram filter
- v45: aug p=0.25 
- v46: aug p=0.3
- v47: base imageもaug強め（scale_lower=0.5>0.2）
- v48: text長を可変に, フォントをランダムに
- v49: base image aug弱め（scale_lower=0.5>0.8）
- v50: v48, jpeq quality p=0.5
- v51: v48, jpeq quality p=0.25
- v52: v51 + colorjitter p=1.0
- v53: v52 + pad & fix
- v54: v52 + scale_lower=0.08
- v55: v52 + scale_lower=0.15
- v56: aug色々調整
- v57: v52 + colorjitter 激しく
- v58: v52 + colorjitter さらに激しく
- v59: MOCO
- v60: DCQ like
- v61: bipartite XBM contrastive
- v62: v58, eff-s
- v63: v58, minerの検証
- v64: v58, eff-s, bs=64, memory_size=20000
- v65: v58, memory_size=10000
- v68: v58のつづき、lr=0.1
- v69: v58のつづき、lr=0.05
- v70: v58のつづき、lr=0.01
- v71: v69のつづき、lr=0.025, bs=64
- v72: v69のつづき、lr=0.025, bs=128
- v73: v58, 別seed


- model1: v58 -> v69 -> v72
- model2: v73 (resumeあり) -> v74 -> v75

python ../scripts/eval_metrics.py v2/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv

### train-subset / eval-subset

python v58.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v58.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v58/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5821057822009852,
  "recall_p90": 0.5069124423963134
}

### train-full / eval-subset
python v58.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v58.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v58/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.6538568969283296,
  "recall_p90": 0.6076938489280705
}

python v73.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 6 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v73.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 6 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --resume v73/train/checkpoint_0001.pth.tar \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v73.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v73/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.6541884424929326,
  "recall_p90": 0.610699258665598
}

### train-full / eval-full

python v58.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v58/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
{
  "average_precision": 0.5850266309735006,
  "recall_p90": 0.47465437788018433
}

lr=0.05から
python v69.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 --lr 0.05 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v58/train/checkpoint_0004.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v69.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v69/train/checkpoint_0004.pth.tar --input-size 384 --target-set qrt ../input/
{
  "average_precision": 0.6179223783734548,
  "recall_p90": 0.5231416549789621
}

python v72.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 777 \
  --epochs 5 --lr 0.025 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v69/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v72.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v72/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v72 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6247096196560734,
  "recall_p90": 0.5237427369264677
}

python v74.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 66 \
  --epochs 5 --lr 0.05 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v73/train/checkpoint_0004.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
gsutil -m cp -r v74 gs://fbisc/exp/
sudo shutdown

python v75.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 666 \
  --epochs 5 --lr 0.025 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v74/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v75.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v75/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v75 gs://fbisc/exp/
sudo shutdown

## ref
https://github.com/facebookresearch/simsiam
