import torch.nn as nn

from encoding_custom import backbones


__all__ = ["BaseNet"]


class BaseNet(nn.Module):
    def __init__(self, name, norm_layer=None, pretrained=True, **kwargs):
        super(BaseNet, self).__init__()
        kwargs.update({'norm_layer': norm_layer})
        self.backbone = getattr(backbones, name, None)
        if self.backbone is None:
            raise ValueError(f'Unknown backbone {name}')
        self.backbone = self.backbone(pretrained=pretrained, **kwargs)

        # TODO: efficientnet and hardnet need to be refactored...
        # They are currently not used...
        # if name.startswith("hardnet"):
        #     self.backbone = {
        #         "hardnet68": partial(backbones.HarDNet, depth_wise=False, arch=68),
        #         "hardnet68d": partial(backbones.HarDNet, depth_wise=True, arch=68),
        #         "hardnet39d": partial(backbones.HarDNet, depth_wise=True, arch=39),
        #         "hardnet85": partial(backbones.HarDNet, depth_wise=False, arch=85),
        #     }[name](pretrained=pretrained)
        # elif "efficientnet" in name:
        #     self.backbone = backbones.efficientnet_pytorch.EfficientNet.from_pretrained(
        #         name)

    def forward(self, x):
        return self.backbone(x)

    def forward_stage(self, x, stage):
        return self.backbone.forward_stage(x, stage)

    def forward_stage_with_last_block(self, x, stage):
        return self.backbone.forward_stage_with_last_block(x, stage)
