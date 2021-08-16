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

python v37.py \
  -a tf_efficientnetv2_l_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 \
  --lr 0.05 --wd 1e-6 \
  --batch-size 96 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 10000 \
  --weight ./v31/train/checkpoint_0004.pth.tar \
  ../input/training_images/
gsutil -m cp -r v37 gs://fbisc/exp/
sudo shutdown

for epoch in `seq 0 4`; do
  python v37.py -a tf_efficientnetv2_l_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v37/train/checkpoint_000${epoch}.pth.tar --input-size 384 --eval-subset ../input/
done
python v37.py \
  -a tf_efficientnetv2_l_in21k \
  --batch-size 256 \
  --mode extract --target-set qrt \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --weight ./v37/train/checkpoint_0004.pth.tar \
  --input-size 384 \
  ../input/

python v38.py \
  -a tf_efficientnetv2_l_in21k \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 777 \
  --epochs 5 \
  --lr 0.001 --wd 1e-6 \
  --batch-size 48 --ncrops 2 \
  --gem-p 3.0 --gem-eval-p 5.0 \
  --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  --weight ./v36/train/checkpoint_0004.pth.tar \
  ../input/training_images/
gsutil -m cp -r v38 gs://fbisc/exp/
sudo shutdown

gsutil -m cp -r gs://fbisc/exp/v38 .
for epoch in `seq 0 4`; do
  python v38.py -a tf_efficientnetv2_l_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p 5.0 --weight ./v38/train/checkpoint_000${epoch}.pth.tar --input-size 512 --eval-subset ../input/
done
for p in {3.0}; do
  python v38.py -a tf_efficientnetv2_l_in21k --batch-size 256 --mode extract --gem-p 3.0 --gem-eval-p ${p} --weight ./v38/train/checkpoint_0004.pth.tar --input-size 512 --eval-subset ../input/
done
python v38.py \
  -a tf_efficientnetv2_l_in21k \
  --batch-size 256 \
  --mode extract --target-set qrt \
  --gem-p 3.0 --gem-eval-p 3.0 \
  --weight ./v38/train/checkpoint_0004.pth.tar \
  --input-size 512 \
  ../input/

python ../scripts/eval_metrics.py v2/extract/fb-isc-submission.h5 ../input/public_ground_truth.csv

### train-subset / eval-subset
python v42.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v42.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v42/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.540043864275826,
  "recall_p90": 0.44980965738328993
}
python v43.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v43.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v43/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5428589187520785,
  "recall_p90": 0.4548186736125025
}
python v44.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v44.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v44/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5381022583159595,
  "recall_p90": 0.453416149068323
}}

python v45.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v45.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v45/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5450802149568448,
  "recall_p90": 0.45762372270086155
}

python v46.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v46.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v46/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5433148524194706,
  "recall_p90": 0.4496092967341214
}

python v47.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v47.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v47/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.540267827410144,
  "recall_p90": 0.44660388699659387
}

python v48.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v48.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v48/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5443596052405804,
  "recall_p90": 0.45742336205169304
}

python v49.py \
  -a tf_efficientnetv2_s_in21k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v49.py -a tf_efficientnetv2_s_in21k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v49/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
0.8: {
  "average_precision": 0.5477583074859417,
  "recall_p90": 0.4568222801041875
}
0.7: {
  "average_precision": 0.5480041266591046,
  "recall_p90": 0.455019034261671
}

python v49.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v49.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v49/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5575901622781466,
  "recall_p90": 0.4670406732117812
}

python v50.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v50.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v50/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.559254542003466,
  "recall_p90": 0.4736525746343418
}

python v51.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v51.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v51/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5601304851367533,
  "recall_p90": 0.47625726307353233
}

python v52.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v52.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v52/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5624654561816684,
  "recall_p90": 0.48146663995191347
}
remove centerCrop
{
  "average_precision": 0.5737274306841692,
  "recall_p90": 0.4976958525345622
}

python v52.py -a tf_efficientnetv2_s_in21ft1k --batch-size 64 --mode extract3 --gem-eval-p 1.0 --weight ./v52/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
python v54.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v54.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v54/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5665281378572695,
  "recall_p90": 0.4890803446203166
}

python v55.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v55.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v55/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5776866687749267,
  "recall_p90": 0.4910839511120016
}

python v56.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v56.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v56/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5631269256010065,
  "recall_p90": 0.48627529553195753
}

python v57.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v57.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v57/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5812654622015935,
  "recall_p90": 0.5067120817471449
}
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

python v60.py -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 --gem-p 1.0 --margin 0.3 --temperature 0.03 --input-size 256 --sample-size 100000 --memory-size 10000 ../input/training_images/
python v60.py -a tf_efficientnetv2_s_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v60/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5860912471974956,
  "recall_p90": 0.48226808254858744
}

python v61.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v61.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v61/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.5834846302748934,
  "recall_p90": 0.4990983770787417
}

set -e
for num_bins in {50,100,200}; do
  for neg_per_bin in {10,20,30}; do
    python v63.py \
      -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
      --epochs 1 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
      --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --num_bins ${num_bins} --neg_per_bin ${neg_per_bin} \
      --input-size 256 --sample-size 100000 --memory-size 10000 \
      ../input/training_images/
    python v63.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v63/train/checkpoint_0000.pth.tar --input-size 256 --eval-subset ../input/
  done
done

num_bins=200
neg_per_bin=20
python v63.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --num_bins ${num_bins} --neg_per_bin ${neg_per_bin} \
  --input-size 256 --sample-size 100000 --memory-size 10000 \
  ../input/training_images/
python v63.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v63/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/

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


python v61.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
python v61.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v61/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.6360787938679936,
  "recall_p90": 0.5822480464836706
}

python v62.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
python v62.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v62/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/
{
  "average_precision": 0.6459662181251028,
  "recall_p90": 0.5998797836104989
}

python v64.py \
  -a tf_efficientnetv2_s_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 15000 \
  ../input/training_images/
python v64.py -a tf_efficientnetv2_s_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v64/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/

python v64.py -a tf_efficientnetv2_s_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v64/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/

python v65.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 7 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 10000 \
  ../input/training_images/
python v65.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v64/train/checkpoint_0004.pth.tar --input-size 256 --eval-subset ../input/

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
v36
{
  "average_precision": 0.5484594023705984,
  "recall_p90": 0.4608294930875576
}
v36 w/ embedding isolation
{
  "average_precision": 0.5920966046321953,
  "recall_p90": 0.5301542776998598
}
v36 w/ embedding isolation & normalization
{
  "average_precision": 0.598628189816694,
  "recall_p90": 0.5305549989981968
}
v36 w/ embedding isolation & normalization & loftr
{
  "average_precision": 0.6301642178240727,
  "recall_p90": 0.5547986375475856
}
v38, p=3
{
  "average_precision": 0.5536742434464547,
  "recall_p90": 0.43418152674814664
}
v38, w/ embedding isolation
{
  "average_precision": 0.6023550401651943,
  "recall_p90": 0.5325586054898818
}
python v58.py -a tf_efficientnetv2_m_in21ft1k --batch-size 512 --mode extract --gem-eval-p 1.0 --weight ./v58/train/checkpoint_0004.pth.tar --input-size 256 --target-set qrt ../input/
{
  "average_precision": 0.5850266309735006,
  "recall_p90": 0.47465437788018433
}

v62
{
  "average_precision": 0.5619927147563503,
  "recall_p90": 0.4241634942897215
}

lr=0.1から
python v68.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v58/train/checkpoint_0004.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v68.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v68/train/checkpoint_0004.pth.tar --input-size 384 --target-set qrt ../input/
{
  "average_precision": 0.6145957668784987,
  "recall_p90": 0.5221398517331196
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

lr=0.01から
python v70.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 77 \
  --epochs 5 --lr 0.01 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v58/train/checkpoint_0004.pth.tar \
  --input-size 384 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v70.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v70/train/checkpoint_0004.pth.tar --input-size 384 --target-set qrt ../input/
{
  "average_precision": 0.6166408668762218,
  "recall_p90": 0.5125225405730315
}

python v71.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 777 \
  --epochs 5 --lr 0.025 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v69/train/checkpoint_0004.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
python v71.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v71/train/checkpoint_0004.pth.tar --input-size 512 --target-set qrt ../input/
{
  "average_precision": 0.6182067154392421,
  "recall_p90": 0.5061109997996394
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


## ref
https://github.com/TengdaHan/ShuffleBN
https://github.com/facebookresearch/simsiam
