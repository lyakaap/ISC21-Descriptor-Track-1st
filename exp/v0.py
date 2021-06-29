from comet_ml import Experiment
import json
import math
import os
import random
from pathlib import Path

import click
import cv2
import editdistance
import numpy as np
import pandas as pd
import timm
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, ParameterGrid, ParameterSampler
from timm.data.auto_augment import rand_augment_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torchvision.io import read_image
from torchvision.transforms import (CenterCrop, ColorJitter, Compose,
                                    Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomVerticalFlip,
                                    Resize, ToTensor)
from tqdm import tqdm

import debug
from src import dataset, utils

ROOT = Path(__file__).absolute().parents[1]
IMG_DIR = ROOT / 'input/bms-molecular-translation/train/'
NUM_WORKERS = os.cpu_count()
SEED = 0

params = {
    'ver': __file__.replace('.py', ''),
    'size': 224,
    'test_size': 224,
    'backbone': 'vit_deit_base_distilled_patch16_224',
    'optimizer': 'adamw',
    'lr': 3e-4,
    'batch_size': 32,
    'epochs': 10,
    'wd': 1e-5,
    'scale_lower': 0.8,
    'scale_upper': 1.0,
    'filter_wd': True,
    'in_chans': 1,
    'layers': 4,
}


class Net(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self):
        pass


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        image_dir: Path,
        transforms,
    ):
        self.dataset = dataset.to_dict("records")
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image_id = self.dataset[i]['image_id']
        image_path = self.image_dir / '/'.join(image_id[:3]) / f'{image_id}.png'
        image = read_image(image_path)
        image = self.transforms(image)
        return image


@click.group()
def cli():
    if not Path(ROOT / f'exp/{params["ver"]}/train').exists():
        Path(ROOT / f'exp/{params["ver"]}/train').mkdir(parents=True)
    if not Path(ROOT / f'exp/{params["ver"]}/tuning').exists():
        Path(ROOT / f'exp/{params["ver"]}/tuning').mkdir(parents=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@cli.command()
@click.option('--tuning', is_flag=True)
@click.option('--dry-run', is_flag=True)
@click.option('--params-path', type=click.Path(), default=None, help='json file path for setting parameters')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
def job(tuning, dry_run, params_path, devices):

    global params
    if tuning:
        with open(params_path, 'r') as f:
            params = json.load(f)
        mode_str = 'tuning'
        setting = '_'.join(
            f'{tp}-{params[tp]}' for tp in params['tuning_params'])
    else:
        mode_str = 'train'
        setting = ''

    exp_path = ROOT / f'exp/{params["ver"]}/'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

    if not dry_run:
        experiment = Experiment(
            api_key=os.environ['COMET_API_KEY'],
            project_name='bms',
            workspace='shuhei-yokoo',
        )
        experiment.set_name(params['ver'] + f'-{setting}' if tuning else '')
        experiment.log_parameters(params)

    logger = utils.get_logger(log_dir=exp_path / f'{mode_str}/log/{setting}')

    train_labels = pd.read_feather('../input/train.feather')
    train_labels['group'] = dataset.create_lengths_groups(lengths=train_labels['token_length'], bins=[75, 90, 110])
    folds = pd.read_csv(F_DATA_DIR / 'bms-kfold/5fold.csv')

    fold = 0

    if dry_run:
        params['epochs'] = 2
        num_samples_per_fold = 1000
        ids_train = folds.loc[folds['fold'] != fold, 'image_id'].sample(num_samples_per_fold * 4).sort_values().values
        ids_valid = folds.loc[folds['fold'] == fold, 'image_id'].sample(num_samples_per_fold).sort_values().values
    else:
        # num_samples_per_fold = 200000
        # ids_train = folds.loc[folds['fold'] != fold, 'image_id'].sample(num_samples_per_fold * 4).sort_values().values
        # ids_valid = folds.loc[folds['fold'] == fold, 'image_id'].sample(num_samples_per_fold).sort_values().values
        ids_train = folds.loc[folds['fold'] != fold, 'image_id'].values
        ids_valid = folds.loc[folds['fold'] == fold, 'image_id'].values

    df_train = train_labels[train_labels['image_id'].isin(ids_train)].reset_index(drop=True)
    df_valid = train_labels[train_labels['image_id'].isin(ids_valid)].reset_index(drop=True)

    transforms = {
        'train': Compose([
            ToTensor(),
            RandomResizedCrop(size=(params['size'], params['size']), scale=(params['scale_lower'], params['scale_upper'])),
            RandomHorizontalFlip(p=0.35),
            RandomVerticalFlip(p=0.35),
        ]),
        'valid': Compose([
            ToTensor(),
            Resize(size=(params['test_size'] + 32, params['test_size'] + 32)),
            CenterCrop((params['test_size'], params['test_size'])),
        ]),
    }
    datasets = {
        'train': ImageCaptioningDataset(df_train, image_dir=IMG_DIR, tokenizer=tokenizer, transforms=transforms['train']),
        'valid': ImageCaptioningDataset(df_valid, image_dir=IMG_DIR, tokenizer=tokenizer, image_transforms=transforms['valid']),
    }
    samplers = {
        'train': RandomSampler(datasets['train']),
        'valid': SequentialSampler(datasets['valid']),
    }
    samplers = {
        'train': dataset.GroupedBatchSampler(sampler=samplers['train'], group_ids=df_train['group'], batch_size=params['batch_size']),
        'valid': dataset.GroupedBatchSampler(sampler=samplers['valid'], group_ids=df_valid['group'], batch_size=params['batch_size'] * 2),
    }
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_sampler=samplers['train'], pin_memory=True, num_workers=NUM_WORKERS, collate_fn=dataset.collate_fn),
        'valid': DataLoader(datasets['valid'], batch_sampler=samplers['valid'], pin_memory=True, num_workers=NUM_WORKERS, collate_fn=dataset.collate_fn),
    }

    optimizer = utils.get_optim(params, model, filter_bias_and_bn=params['filter_wd'], skip_gain=True)
    scheduler = CosineLRScheduler(optimizer, t_initial=params['epochs'])
    scaler = torch.cuda.amp.GradScaler()

    model = model.to('cuda')
    if len(devices.split(',')) > 1:
        model = nn.DataParallel(model)

    best_score = 1e9
    early_stopping_rounds = 2

    for epoch in range(params['epochs']):

        logger.info(
            f'Epoch {epoch}/{params["epochs"]} | lr: {optimizer.param_groups[0]["lr"]}')

        # ============================== train ============================== #
        model.train(True)
        loss_meter = utils.AverageMeter()
        dist_meter = utils.AverageMeter()

        for i, (img, token_ids, gt_inchis) in tqdm(enumerate(data_loaders['train']),
                                                   total=len(data_loaders['train']), miniters=None, ncols=55):
            img = img.to('cuda', non_blocking=True)
            token_ids = token_ids.to('cuda', non_blocking=True)
            with torch.cuda.amp.autocast():
                output = model(img, token_ids)
            loss = output['loss']
            logits = output['logits']
            pred_inchis = tokenizer.decode_batch(logits.max(dim=-1).indices.tolist())
            dist = np.mean([editdistance.eval(pred, gt) for pred, gt in zip(pred_inchis, gt_inchis)])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.size(0))
            dist_meter.update(dist.item(), img.size(0))

            if i % 10 == 9:
                # logger.info(f'pred: {pred_inchis[0]}, gt: {gt_inchis[0]}')
                logger.info(f'[Train] {epoch+i/len(data_loaders["train"]):.2f}epoch |'
                            f'({setting}) loss: {loss_meter.avg:.4f}, dist: {dist_meter.avg:.2f}')

        scheduler.step(epoch=epoch)

        # ============================== eval ============================== #
        model.train(False)
        dists = []
        for i, (img, token_ids, gt_inchis) in tqdm(enumerate(data_loaders['valid']),
                                                   total=len(data_loaders['valid']), miniters=None, ncols=55):
            img = img.to('cuda', non_blocking=True)
            token_ids = token_ids.to('cuda', non_blocking=True)

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model.generate(img)
            # logits = output['logits']
            # pred_inchis = tokenizer.decode_batch(logits.max(dim=-1).indices.tolist())
            pred_inchis = tokenizer.decode_batch(output.cpu().tolist())
            d = [editdistance.eval(pred, gt) for pred, gt in zip(pred_inchis, gt_inchis)]
            dists.extend(d)

            if i % 10 == 9:
                logger.info(f'Example:\ndist: {d[0]:.2f}\npred: {pred_inchis[0]}\ngt: {gt_inchis[0]}')

        dist = np.mean(dists)
        logger.info(f'[Valid] {epoch+1}epoch | ({setting}) dist: {dist:.2f}')
        if not dry_run:
            experiment.log_metric('dist', dist, epoch=epoch)

        if dist < best_score:
            best_score = dist
            last_improved_epoch = epoch
            utils.save_checkpoint(filename=exp_path / f'{mode_str}/log/{setting}/model.pth',
                                  model=model, params=params, epoch=epoch)

        if epoch - last_improved_epoch > early_stopping_rounds:
            logger.info(
                '\n'
                f'early stopping at: {epoch + 1}\n'
                f'best epoch: {last_improved_epoch + 1}\n'
                f'best score: {best_score:.4f}\n'
            )
            break

    if isinstance(model, nn.DataParallel):
        model = model.module

    if tuning:
        tuning_result = {
            'score': best_score,
        }
        utils.write_tuning_result(params, tuning_result, exp_path / 'tuning/results.csv')
    elif not dry_run:
        utils.save_checkpoint(filename=exp_path / f'{mode_str}/log/{setting}/model.pth',
                              model=model, params=params, epoch=epoch)


@cli.command()
@click.option('--mode', type=str, default='grid', help='Search method (tuning)')
@click.option('--n-iter', type=int, default=10, help='n of iteration for random parameter search (tuning)')
@click.option('--n-gpu', type=int, default=-1, help='n of used gpu at once')
@click.option('--devices', '-d', type=str, help='comma delimited gpu device list (e.g. "0,1")')
@click.option('--n-blocks', '-n', type=int, default=1)
@click.option('--block-id', '-i', type=int, default=0)
def tuning(mode, n_iter, n_gpu, devices, n_blocks, block_id):

    if n_gpu == -1:
        n_gpu = len(devices.split(','))

    space = [
        {
            'lr': [3e-4],
            'layers': [3],
            'batch_size': [24],
            # 'optimizer': ['momentum', 'nesterov'],
        },
    ]

    if mode == 'grid':
        candidate_list = list(ParameterGrid(space))
    elif mode == 'random':
        candidate_list = list(ParameterSampler(
            space, n_iter, random_state=SEED))
    else:
        raise ValueError

    n_per_block = math.ceil(len(candidate_list) / n_blocks)
    candidate_chunk = candidate_list[block_id *
                                     n_per_block: (block_id + 1) * n_per_block]

    utils.launch_tuning(mode=mode, n_iter=n_iter, n_gpu=n_gpu, devices=devices,
                        params=params, root=ROOT, candidate_list=candidate_chunk)


if __name__ == '__main__':
    cli()
