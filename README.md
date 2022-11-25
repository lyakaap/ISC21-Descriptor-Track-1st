# ISC21-Descriptor-Track-1st

The 1st Place Solution of the Facebook AI Image Similarity Challenge (ISC21) : Descriptor Track.

You can check our solution tech report from: [Contrastive Learning with Large Memory Bank and Negative Embedding Subtraction for Accurate Copy Detection](https://arxiv.org/abs/2112.04323)

## Installation

```
pip install git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st
```

## Usage

```
import requests
import torch
from PIL import Image

from isc_feature_extractor import create_model

recommended_weight_name = 'isc_ft_v107'
model, preprocessor = create_model(weight_name=recommended_weight_name, device='cpu')

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
x = preprocessor(image).unsqueeze(0)

y = model(x)
print(y.shape)  # => torch.Size([1, 256])
```
