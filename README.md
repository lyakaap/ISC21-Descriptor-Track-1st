# Facebook AI Image Similarity Challenge

## setup

### OS
Ubuntu 18.04

### CUDA Version
11.1

### environment

Run this for python env

```
conda env create -f environment.yml
```

### data download

```
mkdir -p input/{query,reference,train}_images
aws s3 cp s3://drivendata-competition-fb-isc-data/all/query_images/ input/query_images/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data/all/reference_images/ input/reference_images/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data/all/train_images/ input/train_images/ --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data/all/query_images_phase2/ input/query_images_phase2/ --recursive --no-sign-request
```

## train

Run below lines step by step.

```
cd exp

CUDA_VISIBLE_DEVICES=0,1,2,3 python v83.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 9 \
  --epochs 5 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
CUDA_VISIBLE_DEVICES=0,1,2,3 python v83.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 90 \
  --epochs 10 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 256 --sample-size 1000000 --memory-size 20000 \
  --resume ./v83/train/checkpoint_0004.pth.tar \
  ../input/training_images/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python v86.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99 \
  --epochs 7 --lr 0.1 --wd 1e-6 --batch-size 128 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 \
  --input-size 384 --sample-size 1000000 --memory-size 20000 --weight ./v83/train/checkpoint_0005.pth.tar \
  ../input/training_images/

python v98.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999 \
  --epochs 3 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v86/train/checkpoint_0005.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/

python v107.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.5 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v98/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/
```

The final model weight can be downloaded from here: https://drive.google.com/file/d/1ySea-NJp_J0aWvma_WmVbc3Hnwf5LHUf/view?usp=sharing
You can execute inference code without run training with this model weight.
To locate the model weight to suitable location, run following commands after downloaded the model weight.

```
mkdir -p exp/v107/train
mv checkpoint_009.pth.tar exp/v107/train/
```

## inference

Note that faiss doesn't work with A100, so I used 4x GTX 1080 Ti for post-process.

```
cd exp

python v107.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v107/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/

# this script generates final prediction result files
python ../scripts/postprocess.py
```

Submission files are outputted here:

- `exp/v107/extract/v107_iso.h5`  # descriptor track
- `exp/v107/extract/v107_iso.csv`  # matching track

descriptor track local evaluation score:

```
{
  "average_precision": 0.9479039085717805,
  "recall_p90": 0.9192546583850931
}
```

matching track local evaluation score:

```
{
  "average_precision": 0.9479246965365696,
  "recall_p90": 0.9192546583850931
}
```

## after

```
python v86.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v86/train/checkpoint_0005.pth.tar --input-size 384 --target-set qrt ../input/

python v207.py -a tf_efficientnetv2_m_in21ft1k --batch-size 256 --mode extract --gem-eval-p 1.0 --weight ./v207/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
```

```
gsutil -m cp -r gs://fbisc/exp/v86 .
python v198.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 999 \
  --epochs 3 --lr 0.1 --wd 1e-6 --batch-size 64 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.0 --weight ./v86/train/checkpoint_0005.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 20000 \
  ../input/training_images/
`
python v207.py \
  -a tf_efficientnetv2_m_in21ft1k --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 99999 \
  --epochs 10 --lr 0.5 --wd 1e-6 --batch-size 16 --ncrops 2 \
  --gem-p 1.0 --pos-margin 0.0 --neg-margin 1.1 --weight ./v198/train/checkpoint_0001.pth.tar \
  --input-size 512 --sample-size 1000000 --memory-size 1000 \
  ../input/training_images/

python v207.py -a tf_efficientnetv2_m_in21ft1k --batch-size 128 --mode extract --gem-eval-p 1.0 --weight ./v207/train/checkpoint_0009.pth.tar --input-size 512 --target-set qrt ../input/
```
