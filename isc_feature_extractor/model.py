from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

AVAILABLE_MODELS = {
    "isc_selfsup_v98": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_selfsup_v98.pth.tar",
    "isc_ft_v107": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
}
DEFAULT_CKPT_PATH = torch.hub.get_dir()


class ISCNet(nn.Module):
    """
    Feature extractor for image copy-detection task.

    Args:
        backbone (`nn.Module`):
            Backbone module.
        fc_dim (`int=256`):
            Feature dimension of the fc layer.
        p (`float=1.0`):
            Power used in gem pooling for training.
        eval_p (`float=1.0`):
            Power used in gem pooling for evaluation. In practice, using a larger power
            for evaluation than training can yield a better performance.
    """

    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        p: float = 1.0,
        eval_p: float = 1.0,
        l2_normalize=True,
    ):

        super().__init__()

        self.backbone = backbone
        self.fc = nn.Linear(
            self.backbone.feature_info.info[-1]["num_chs"], fc_dim, bias=False
        )
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p
        self.l2_normalize = l2_normalize

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        p = self.p if self.training else self.eval_p
        x = gem(x, p).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x)
        return x


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def create_model(
    weight_name: str = "isc_ft_v107",
    model_dir: str = DEFAULT_CKPT_PATH,
    fc_dim: int = 256,
    p: float = 1.0,
    eval_p: float = 1.0,
    l2_normalize: bool = True,
    device: str = "cuda",
    is_training: bool = False,
) -> tuple[ISCNet, transforms.Compose]:
    """
    Create a model for image copy-detection task.

    Args:
        weight_name (`str=None`):
            Weight name. If None, use the default weight.
            Available weights are:
                - `isc_selfsup_v98`: Self-supervised pre-trained model.
                - `isc_ft_v107`: Fine-tuned model using ISC ground truth data.
        model_dir (`str=DEFAULT_CKPT_PATH`):
            Directory to save the default weight.
        fc_dim (`int=256`):
            Feature dimension of the fc layer.
        p (`float=1.0`):
            Power used in gem pooling for training.
        eval_p (`float=1.0`):
            Power used in gem pooling for evaluation.
        l2_normalize (`bool=True`):
            Whether to normalize the feature vector.
        device (`str='cuda'`):
            Device to load the model.
        is_training (`bool=False`):
            Whether to load the model for training.

    Returns:
        model:
            ISCNet model.
        preprocessor:
            Preprocess function tied to model.
    """
    if not weight_name in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid weight name: {weight_name}, "
            f"available weights are: {AVAILABLE_MODELS.keys()}"
        )

    ckpt = torch.hub.load_state_dict_from_url(
        AVAILABLE_MODELS[weight_name],
        model_dir=model_dir,
        map_location="cpu",
    )
    arch = ckpt["arch"]  # tf_efficientnetv2_m_in21ft1k
    input_size = ckpt["args"].input_size

    backbone = timm.create_model(arch, features_only=True)
    model = ISCNet(
        backbone=backbone,
        fc_dim=fc_dim,
        p=p,
        eval_p=eval_p,
        l2_normalize=l2_normalize,
    )
    model.to(device).train(is_training)

    state_dict = {}
    for s in ckpt["state_dict"]:
        state_dict[s.replace("module.", "")] = ckpt["state_dict"][s]

    if fc_dim != 256:
        # interpolate to new fc_dim
        state_dict["fc.weight"] = F.interpolate(
            state_dict["fc.weight"].permute(1, 0).unsqueeze(0),
            size=fc_dim, mode="linear", align_corners=False,
        ).squeeze(0).permute(1, 0)
        for bn_param in ["bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]:
            state_dict[bn_param] = F.interpolate(
                state_dict[bn_param].unsqueeze(0).unsqueeze(0),
                size=fc_dim, mode="linear", align_corners=False,
            ).squeeze(0).squeeze(0)

    model.load_state_dict(state_dict)

    preprocessor = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=backbone.default_cfg["mean"],
                std=backbone.default_cfg["std"],
            ),
        ]
    )

    return model, preprocessor
