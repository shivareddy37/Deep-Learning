import torch
import torch.nn as nn
from midas.base_model import BaseModel
from midas.blocks import (
    Interpolate,
    _make_encoder,
    FeatureFusionBlock_custom,
)
from midas.backbones.livit import stem_b4_transpose, forward_levit
from midas.backbones.vit import forward_vit
from midas.backbone.swin_common import forward_swin
from timm.models.layers import get_act_layers


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            "swin2l24_384": [1, 1, 17, 1],  # Allowed ranges: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        if "swin" in backbone:
            self.forward_transformer = forward_swin
        else:
            self.forward_transformer = forward_vit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)