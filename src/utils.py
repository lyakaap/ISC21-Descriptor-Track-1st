from sklearn.decomposition import PCA
import json
import logging
import random
import os
import subprocess
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, ParameterSampler
from torch.utils.tensorboard import SummaryWriter
from .optimizer import SAM, MADGRAD


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WhiteningPCA:
    def __init__(self, eps=1e-9):
        self.eps = eps
        self.m = None
        self.P = None

    def fit(self, X: np.ndarray):
        """size should be N >> D"""
        N, D = X.shape

        self.m = X.mean(axis=0)  # (D,)
        X_centered = X - self.m
        C = X_centered.transpose() @ X_centered / N  # (D, D)

        S, U = np.linalg.eig(C)
        order = S.argsort()[::-1]
        S = S[order] + self.eps
        U = U[order]

        self.P = np.diag(S ** -0.5) @ U  # (D, D)

    def transform(self, X: np.ndarray, n_components=None):
        X_whitened = (X - self.m) @ self.P.transpose()

        if n_components is not None:
            X_whitened = X_whitened[:, :n_components]

        return X_whitened


def apply_whitening_pca(X, n_components=None):
    pca = PCA(whiten=True)
    pca.fit(X)
    m = pca.mean_
    P = pca.components_.T / np.sqrt(pca.explained_variance_)

    X_whitened = (X - m) @ P.transpose()
    if n_components is not None:
        X_whitened = X_whitened[:, :n_components]

    return X_whitened


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if 'module' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # str.replace() can't be used because of unintended key removal (e.g. se-module)
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(filename, model, optimizer=None, params=None, epoch=None):
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    attributes = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'params': params,
        'epoch': epoch,
    }
    torch.save(attributes, filename)


def get_logger(log_dir, loglevel=logging.INFO, tensorboard_dir=None):
    import logzero

    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.logfile(log_dir / 'logfile')

    if tensorboard_dir is not None:
        if not Path(tensorboard_dir).exists():
            Path(tensorboard_dir).mkdir(parents=True)
        writer = SummaryWriter(tensorboard_dir)

        return logzero.logger, writer

    return logzero.logger


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def add_weight_decay2(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or "gain" in name or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_optim(params, model, filter_bias_and_bn=False, skip_list=(), skip_gain=False):

    if params['optimizer'] == 'adamw' or params['optimizer'] == 'radam':
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        # params['wd'] /= params['lr']
        pass
    if params['wd'] > 0 and filter_bias_and_bn:
        if skip_gain:
            target = add_weight_decay2(model, params['wd'], skip_list=skip_list)
        else:
            target = add_weight_decay(model, params['wd'], skip_list=skip_list)
        weight_decay = 0.
    else:
        target = model.parameters()
        weight_decay = params['wd']

    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(target, params['lr'], weight_decay=weight_decay)
    elif params['optimizer'] == 'momentum':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9, weight_decay=weight_decay)
    elif params['optimizer'] == 'nesterov':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(target, params['lr'], weight_decay=weight_decay, eps=1e-7)
    elif params['optimizer'] == 'amsgrad':
        optimizer = optim.Adam(target, params['lr'], weight_decay=weight_decay, amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(target, params['lr'], weight_decay=weight_decay, eps=1e-7)
    elif params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(target, params['lr'], weight_decay=weight_decay, eps=1e-7)
    elif params['optimizer'] == 'radam':
        from timm.optim import RAdam
        optimizer = RAdam(target, params['lr'], weight_decay=weight_decay)
    elif params['optimizer'] == 'nadam':
        from timm.optim import Nadam
        optimizer = Nadam(target, params['lr'], weight_decay=weight_decay)
    elif params['optimizer'] == 'novograd':
        from timm.optim import NovoGrad
        optimizer = NovoGrad(target, lr=params['lr'], weight_decay=weight_decay)
    elif params['optimizer'] == 'sam':
        base_optimizer = optim.SGD
        optimizer = SAM(target, base_optimizer, lr=params['lr'], weight_decay=weight_decay, momentum=0.9)
    elif params['optimizer'] == 'madgrad':
        optimizer = MADGRAD(target, lr=params['lr'], weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError

    return optimizer


def write_tuning_result(params: dict, result: dict, df_path: str) -> None:
    row = pd.DataFrame()
    for key in params['tuning_params']:
        row[key] = [params[key]]

    for key, val in result.items():
        row[key] = val

    import lockfile
    with lockfile.FileLock(df_path):
        df_results = pd.read_csv(df_path)
        df_results = pd.concat([df_results, row], sort=False).reset_index(drop=True)
        df_results.to_csv(df_path, index=None)


def check_duplicate(df: pd.DataFrame, p: dict):
    """check if current params combination has already done"""

    new_key_is_included = not all(map(lambda x: x in df.columns, p.keys()))
    if new_key_is_included:
        return False

    for i in range(len(df)):  # for avoiding unexpected cast due to row-slicing
        is_dup = True
        for key, val in p.items():
            if df.loc[i, key] != val:
                is_dup = False
                break
        if is_dup:
            return True
    else:
        return False


def launch_tuning(mode: str = 'grid', n_iter: int = 1, n_gpu: int = 1, devices: str = '0',
                  params: dict = None, space: dict = None, root: str = '../', candidate_list=None):
    """
    Launch paramter search by specific way.
    Each trials are launched asynchronously by forking subprocess and all results of trials
    are automatically written in csv file.

    :param mode: the way of parameter search, one of 'grid or random'.
    :param n_iter: num of iteration for random search.
    :param n_gpu: num of gpu used at one trial.
    :param devices: gpu devices for tuning.
    :param params: training parameters.
                   the values designated as tuning parameters are overwritten
    :param space: paramter search space.
    :param root: path of the root directory.
    """

    gpu_list = deque(devices.split(','))

    if candidate_list is None:
        if mode == 'grid':
            candidate_list = list(ParameterGrid(space))
        elif mode == 'random':
            candidate_list = list(ParameterSampler(space, n_iter))
        else:
            raise ValueError

    params['tuning_params'] = list(candidate_list[0].keys())

    df_path = root / f'exp/{params["ver"]}/tuning/results.csv'
    if Path(df_path).exists() and Path(df_path).stat().st_size > 5:
        df_results = pd.read_csv(df_path)
    else:
        cols = list(candidate_list[0].keys())
        df_results = pd.DataFrame(columns=cols)
        df_results.to_csv(df_path, index=False)

    procs = []
    for p in candidate_list:

        if check_duplicate(df_results, p):
            print(f'skip: {p} because this setting is already experimented.')
            continue

        # overwrite hyper parameters for search
        for key, val in p.items():
            params[key] = val

        while True:
            if len(gpu_list) >= n_gpu:
                devices = ','.join([gpu_list.pop() for _ in range(n_gpu)])
                setting = '_'.join(f'{key}-{val}' for key, val in p.items())
                params_path = root / f'exp/{params["ver"]}/tuning/params_{setting}.json'
                with open(params_path, 'w') as f:
                    json.dump(params, f)
                break
            else:
                time.sleep(1)
                for i, (proc, dev) in enumerate(procs):
                    if proc.poll() is not None:
                        gpu_list += deque(dev.split(','))
                        del procs[i]

        cmd = f'{sys.executable} {params["ver"]}.py job ' \
              f'--tuning --params-path {params_path} --devices "{devices}"'
        procs.append((subprocess.Popen(cmd, shell=True), devices))

    while True:
        time.sleep(1)
        if all(proc.poll() is not None for i, (proc, dev) in enumerate(procs)):
            print('All parameter combinations have finished.')
            break

    show_tuning_result(params["ver"])


def show_tuning_result(ver, mode='markdown', sort_by=None, ascending=False):

    table = pd.read_csv(f'../exp/{ver}/tuning/results.csv')
    if sort_by is not None:
        table = table.sort_values(sort_by, ascending=ascending)

    if mode == 'markdown':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='pipe', showindex=False))
    elif mode == 'latex':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='latex', floatfmt='.2f', showindex=False))
    else:
        from IPython.core.display import display
        display(table)
