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
- v73,v74,v75: v58, 別seed
- v76: v58, backbone tuning
- v77: HybridEmbed, size=224
- v78: HybridEmbed, size=384
- v79: overlay image again
- v80: DOLG, TTA codeつき
- v81: DOLG, simplified
- v82: DOLG, lr fixed
- v83: v79, lr fixed, lr=0.1
- v84: v83, increase, lr=0.02
- v85: v84, increase, lr=0.004
- v86: v83, increase, lr=0.1
- v87: v86, increase, lr=0.1, lr decay
- v88: 最初から512x512
- v89: query-reference pairを学習
- v90: v79, lr=0.05, 384
- v91: v90, lr-0.025, 512
- v92: v84, lr=0.01, fixed, 512
- v93: DOLG, 512
- v94: v84, query-reference pairを学習, res=384
- v95: v84, query-reference pairを学習, res=512
- v96: v94, neg_margin=1.0
- v97: v95, neg_margin=1.0
- v98: train with ref(negative) finetuning
- v99: train with ref(negative) finetuning, neg1.1, ep2
- v100: v98 -> query-reference pairを学習
- v101: v98 -> query-reference pairを学習 with neg ref
- v102: v98 -> query-reference pairを学習 with neg ref, x8
- v103: v98 -> query-reference pairを学習 with neg ref, x16
- v104: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.2
- v105: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.3
- v106: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.4
- v106: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.4
- v106: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.4
- v106: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.4
- v110: v98 -> query-reference pairを学習 with neg ref, x32, lr=0.4


- query trainingをv84からinput_res=512でやる。

- model1: v58 -> v69 -> v72
- model2: v73 (resumeあり) -> v74 -> v75
- model3: v83 -> v84 -> v85
- model4: v83 -> v86 -> v87 -> v89
- model5: v88
- model6: v79 -> v90 -> v91
- model7: v83 -> v84 -> v92
- model8: v83 -> v86 -> v98 -> v100

python ../scripts/eval_metrics.py v2/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv

- cat -> norm -> pca -> norm -> iso -> eval:
{
  "average_precision": 0.6693173312873409,
  "recall_p90": 0.5944700460829493
}


{
  "average_precision": 0.7714079030667595,
  "recall_p90": 0.7295131236225205
}
{
  "average_precision": 0.7751591024253895,
  "recall_p90": 0.7303145662191945
}


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

for pos in 0.0 0.1 0.2; do
  for neg in 0.8 0.9 1.0 1.1; do
    echo $pos $neg
    python v76.py \
      -a dm_nfnet_f0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
      --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
      --gem-p 1.0 --pos-margin $pos --neg-margin $neg \
      --input-size 256 --sample-size 100000 --memory-size 10000 \
      ../input/training_images/
    python v76.py -a dm_nfnet_f0 --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v76/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
  done
done

dm_nfnet_f0
{
  "average_precision": 0.5978324371391083,
  "recall_p90": 0.5193348026447606
}

python v77.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 8 \
  --epochs 5 --lr 1e-3 --wd 1e-2 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 0.8 \
  --input-size 224 --sample-size 100000 --memory-size 1000 \
  ../input/training_images/
python v77.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v77/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/

python v80.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v80.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v80/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/
{
  "average_precision": 0.5906854733441096,
  "recall_p90": 0.5209376878381086
}

python v80.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 3.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v80.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 3.0 --weight ./v80/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/
{
  "average_precision": 0.570341254409829,
  "recall_p90": 0.5003005409737528
}

python v80.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 5000 \
  ../input/training_images/
python v80.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v80/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/
{
  "average_precision": 0.5781064103026176,
  "recall_p90": 0.5037066720096173
}

python v81.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v81.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v81/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/
{
  "average_precision": 0.588532128482479,
  "recall_p90": 0.517130835503907
}

python v84.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 224 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v84.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v84/train/checkpoint_0004.pth.tar --input-size 224 --eval-subset ../input/

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

python v75.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 666 \
  --epochs 5 --lr 0.025 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v74/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v75.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v75/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v75 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6305097451879395,
  "recall_p90": 0.5263474253656582
}

python v76.py \
  -a dm_nfnet_f0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 8 \
  --epochs 5 --lr 0.01 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v76.py -a dm_nfnet_f0 --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v76/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
{
  "average_precision": 0.5633122016307757,
  "recall_p90": 0.454217591664997
}
python v76.py \
  -a dm_nfnet_f0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 8 \
  --epochs 5 --lr 0.05 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
python v76.py -a dm_nfnet_f0 --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v76/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/

gsutil -m cp -r v76 gs://fbisc/exp/
sudo shutdown

python v79.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v79.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v79/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
gsutil -m cp -r v79 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.5895739548539529,
  "recall_p90": 0.49368863955119213
}

python v80.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v80.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v80/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
gsutil -m cp -r v80 gs://fbisc/exp/
sudo shutdown

python v82.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v82.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v82/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
gsutil -m cp -r v82 gs://fbisc/exp/
sudo shutdown

for epoch in `seq 0 6`;
do
  python v84.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v84/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done
for epoch in `seq 0 4`;
do
  python v85.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v85/train/checkpoint_000${epoch}.pth.tar --input-size 512 --eval-subset ../input/
done
for epoch in `seq 4 6`;
do
  python v86.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v86/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done
for epoch in `seq 6 9`;
do
  python v88.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v88/train/checkpoint_000${epoch}.pth.tar --input-size 512 --eval-subset ../input/
done
for epoch in `seq 4 6`;
do
  python v93.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v93/train/checkpoint_000${epoch}.pth.tar --input-size 512 --eval-subset ../input/
done
for epoch in `seq 0 0`;
do
  python v98.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v98/train/checkpoint_000${epoch}.pth.tar --input-size 512 --eval-subset ../input/
done
python v82.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v82/train/checkpoint_0001.pth.tar --input-size 256 --target-set qr ../input/
{
  "average_precision": 0.5570422225858732,
  "recall_p90": 0.4337808054498097
}
0001.pthが最高
プラン
- 最初から最大サイズでlr decayあり
- lr-fixで2epoch->2epoch->2epochとか
  - そのときstepごとにdecayさせる。また、2epochだけじゃなく、ちょっと長めに学習させる。（長い分には困らないので）


python v83.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v83.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 90 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  --resume ./v83/train/checkpoint_0004.pth.tar \
  ../input/training_images/
python v83.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v83/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
gsutil -m cp -r v83 gs://fbisc/exp/

python v84.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99 \
  --epochs 7 --lr 0.02 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  --weight ./v83/train/checkpoint_0005.pth.tar \
  ../input/training_images/
gsutil -m cp -r v84 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6266863798687519,
  "recall_p90": 0.5249449008214787
}

python v85.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 777 \
  --epochs 5 --lr 0.01 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v85.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v85/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v85 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6225837253337314,
  "recall_p90": 0.517531556802244
}

python v86.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99 \
  --epochs 7 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 20000 --weight ./v83/train/checkpoint_0005.pth.tar \
  ../input/training_images/
gsutil -m cp -r v86 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6151627745675662,
  "recall_p90": 0.49789621318373073
}

python v87.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v86/train/checkpoint_0005.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v87.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v87/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v87 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6216224047514121,
  "recall_p90": 0.5043077539571228
}

python v88.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v88.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v88/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v88 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6161563142445075,
  "recall_p90": 0.5261470647164896
}


python v89.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v87/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v89.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v89/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
{
  "average_precision": 0.7714934941398898,
  "recall_p90": 0.7295131236225205
}
python v89.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v89/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.7004156612605895,
  "recall_p90": 0.5832498497295131
}
emb-iso
{
  "average_precision": 0.7247497686934267,
  "recall_p90": 0.6609897816068924
}
tta: TBD
python v89.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v89/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt --tta ../input/


python v84.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v84/train/checkpoint_0006.pth.tar --input-size 384 --target-set qr ../input/
python v86.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v86/train/checkpoint_0005.pth.tar --input-size 384 --target-set qr ../input/
python v90.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v90/train/checkpoint_0004.pth.tar --input-size 384 --target-set qr ../input/

python v90.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 1010 \
  --epochs 5 --lr 0.05 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  --weight ./v79/train/checkpoint_0004.pth.tar \
  ../input/training_images/
gsutil -m cp -r v90 gs://fbisc/exp/
sudo shutdown
{
  "average_precision": 0.6226625143662813,
  "recall_p90": 0.5247445401723102
}

python v91.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 101010 \
  --epochs 5 --lr 0.025 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v90/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v91.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v91/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v91 gs://fbisc/exp/
sudo shutdown

python v92.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 10101010 \
  --epochs 5 --lr 0.01 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v92.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v92/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
gsutil -m cp -r v92 gs://fbisc/exp/
sudo shutdown

python v93.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 11 \
  --epochs 7 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v93.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v93/train/checkpoint_0006.pth.tar --input-size 512 --eval-subset ../input/

python v94.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 256 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v94.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v94/train/checkpoint_0009.pth.tar --input-size 384 --eval-subset ../input/
{
  "average_precision": 0.7361046194356035,
  "recall_p90": 0.692246042877179
}
python v95.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9999999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v95.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v95/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
{
  "average_precision": 0.7516081849416207,
  "recall_p90": 0.7026647966339411
}
python v96.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 256 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v96.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v96/train/checkpoint_0009.pth.tar --input-size 384 --eval-subset ../input/
python v97.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9999999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v84/train/checkpoint_0006.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v97.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v97/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/

python v94.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v94/train/checkpoint_0009.pth.tar --input-size 384 --target-set qrt ../input/
python v95.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v95/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
python v96.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v96/train/checkpoint_0009.pth.tar --input-size 384 --target-set qrt ../input/
python v97.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v97/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/

<!-- python v98.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999999 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v87/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v98.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v98/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/ -->

python v98.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999 \
  --epochs 3 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v86/train/checkpoint_0005.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
gsutil -m cp -r v98 gs://fbisc/exp/
sudo shutdown
python v98.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v98/train/checkpoint_0002.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.6499580591524714,
  "recall_p90": 0.572831095972751
}

python v98.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v98/train/checkpoint_0001.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.6519199107037544,
  "recall_p90": 0.5710278501302344
}

python v99.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999 \
  --epochs 2 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v86/train/checkpoint_0005.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
gsutil -m cp -r v99 gs://fbisc/exp/
python v99.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v99/train/checkpoint_0000.pth.tar --input-size 512 --target-set qr ../input/
python v99.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v99/train/checkpoint_0001.pth.tar --input-size 512 --target-set qr ../input/

python v100.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v100.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v100/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
{
  "average_precision": 0.7618338112257139,
  "recall_p90": 0.7200961731116009
}
python v100.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v100/train/checkpoint_0009.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.7007949228921966,
  "recall_p90": 0.6131035864556201
}

python v101.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v101.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v101/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v101.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v101/train/checkpoint_0009.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.788527565890466,
  "recall_p90": 0.7501502704868763
}

python v102.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 32 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v102.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v102/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v102.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v102/train/checkpoint_0009.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.8144373167650504,
  "recall_p90": 0.7796032859146463
}

python v103.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v103.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v103/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v103.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v103/train/checkpoint_0009.pth.tar --input-size 512 --target-set qr ../input/
{
  "average_precision": 0.8559403478755391,
  "recall_p90": 0.820476858345021
}
{
  "average_precision": 0.7932400790392375,
  "recall_p90": 0.6888399118413143
}

python v104.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.2 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v104.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v104/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v104.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v104/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.9083907640116416,
  "recall_p90": 0.8859947906231216
}
{
  "average_precision": 0.8567829369192675,
  "recall_p90": 0.757964335804448
}

python v105.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.3 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v105.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v105/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v105.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v105/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.9363638667723998,
  "recall_p90": 0.9196553796834301
}
{
  "average_precision": 0.8944961149903653,
  "recall_p90": 0.81526748146664
}


python v106.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.4 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v106.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v106/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
{
  "average_precision": 0.9536196758022021,
  "recall_p90": 0.939491083951112
}
{
  "average_precision": 0.9211411474436273,
  "recall_p90": 0.8659587257062713
}
emb-iso
{
  "average_precision": 0.9306235035379359,
  "recall_p90": 0.9028250851532759
}

csv
{
  "average_precision": 0.9212168067580401,
  "recall_p90": 0.8659587257062713
}


python v104.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v104/train/checkpoint_0009.pth.tar --input-size 512 --target-set t ../input/
python v105.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v105/train/checkpoint_0009.pth.tar --input-size 512 --target-set t ../input/
python v106.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v106/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/

python v107.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.5 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v107.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v107/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v107.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v107/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.9676346030079946,
  "recall_p90": 0.9541174113404127
}
{
  "average_precision": 0.9399111736066676,
  "recall_p90": 0.8964135443798837
}

python v108.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.7 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v108.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v108/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v108.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v108/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.9803096783163888,
  "recall_p90": 0.9729513123622521
}

python v109.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 1.0 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v109.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v109/train/checkpoint_0009.pth.tar --input-size 512 --eval-subset ../input/
python v109.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v109/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.9903292967079874,
  "recall_p90": 0.9861751152073732
}

python v110.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 1.0 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 640 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
python v110.py -a tf_efficientnetv2_m_in21ft1k --batch-size 640 --mode extract --gem-eval-p 1.0 --weight ./v110/train/checkpoint_0009.pth.tar --input-size 640 --eval-subset ../input/
python v110.py -a tf_efficientnetv2_m_in21ft1k --batch-size 640 --mode extract --gem-eval-p 1.0 --weight ./v110/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/

## ref
https://github.com/facebookresearch/simsiam
