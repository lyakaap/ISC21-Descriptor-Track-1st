#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image, ImageFilter
from tqdm import tqdm

import timm


# In[3]:


timm.list_models('*nfnet*')


# In[13]:


model = timm.create_model('dm_nfnet_f0', features_only=True, pretrained=True)

import torch.nn.functional as F
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class ISCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        transforms,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        image = self.transforms(image)
        return image


# In[18]:


from types import SimpleNamespace
args = SimpleNamespace()

args.data = '../input/'
args.batch_size = 128
args.workers = os.cpu_count()


# In[ ]:


query_paths = sorted(Path(args.data).glob('query_images/**/*.jpg'))
query_ids = np.array([p.stem for p in query_paths], dtype='S6')

reference_paths = sorted(Path(args.data).glob('reference_images/**/*.jpg'))
reference_ids = np.array([p.stem for p in reference_paths], dtype='S7')

model.eval().cuda()

cudnn.benchmark = True


# In[53]:


preprocesses = [
    transforms.Resize((384, 384)),
    # transforms.Resize(model.default_cfg['input_size'][1:]),
    # transforms.Resize(model.default_cfg['input_size'] + 32),
    # transforms.CenterCrop(model.default_cfg['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std']),
]

datasets = {
    'query': ISCDataset(query_paths, transforms.Compose(preprocesses)),
    'reference': ISCDataset(reference_paths, transforms.Compose(preprocesses)),
}
loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
data_loaders = {
    'query': torch.utils.data.DataLoader(datasets['query'], **loader_kwargs),
    'reference': torch.utils.data.DataLoader(datasets['reference'], **loader_kwargs),
}

def calc_feats(loader):
    feats = []
    for image in tqdm(loader, total=len(loader)):
        x = image.cuda()
        with torch.no_grad():
            y = model(x)[-1]
            y = gem(y).squeeze(-1).squeeze(-1)
        feats.append(y.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    feats /= np.linalg.norm(feats, 2, axis=1, keepdims=True)
    return feats.astype(np.float32)

query_feats = calc_feats(data_loaders['query'])
reference_feats = calc_feats(data_loaders['reference'])

out = f'fb-isc-submission.h5'
with h5py.File(out, 'w') as f:
    f.create_dataset('query', data=query_feats)
    f.create_dataset('reference', data=reference_feats)
    f.create_dataset('query_ids', data=query_ids)
    f.create_dataset('reference_ids', data=reference_ids)


# In[ ]:




