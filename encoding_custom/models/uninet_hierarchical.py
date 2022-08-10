from functools import partial
from collections import OrderedDict
import torch.nn as nn

from encoding_custom.backbones.base import BaseNet
from encoding_custom.models.model_utils import *
from utilities import generic_utils


class UninetHierarchical(nn.Module):
    def __init__(self, backbone, tasks, norm_layer, cfg=None, pretrained=True,
                 **kwargs):
        super(UninetHierarchical, self).__init__()

        encoder_type = cfg.MODEL.ENCODER.ENCODER_TYPE
        decoder_type = cfg.MODEL.DECODER.DECODER_TYPE

        # only considering semantic seg, depth, semantic cont, surface normals.
        create_basenet = partial(BaseNet, backbone, pretrained=pretrained,
                                 norm_layer=norm_layer, **cfg.BACKBONE_KWARGS)
        self.base_net = create_basenet()
        backbone_feat_channels = self.base_net.backbone.feat_channels

        self.encoder_decoder = UninetEncoderDecoder(
            backbone_feat_channels, create_basenet, encoder_type, decoder_type, tasks,
            norm_layer, cfg, **kwargs)
        kwargs.update({'en_feat_channels': self.encoder_decoder.en_feat_channels})

        self.head = UninetHead(tasks, cfg, norm_layer, **kwargs)

    def forward(self, x, targets=None):
        image_size = list(x.shape[2:])
        backbone_features = self.base_net(x)
        task_to_de_feats, task_to_en_feats, acts = self.encoder_decoder(
            backbone_features)

        out = self.head(task_to_de_feats, task_to_en_feats, image_size)
        out.update({'activations': acts})
        return out


class UninetEncoderDecoder(nn.Module):
    def __init__(self, backbone_feat_channels, create_basenet, encoder_type,
                 decoder_type, tasks, norm_layer, cfg, **kwargs):
        super(UninetEncoderDecoder, self).__init__()

        create_encoder = partial(
            get_encoder, encoder_type, norm_layer, backbone_feat_channels, True, cfg)
        self.uninet_encoder, uni_en_feat_channels = create_encoder()

        en_feat_channels = backbone_feat_channels + uni_en_feat_channels
        self.en_feat_channels = en_feat_channels[:cfg.MODEL.ENCODER.NUM_EN_FEATURES]
        create_decoder = partial(
            get_decoder, decoder_type, norm_layer, self.en_feat_channels, cfg)

        num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES
        self.task_to_en_block = {
            task: block for task, block in dict(cfg.TASK_TO_EN_BLOCK).items()
            if task in tasks}

        self.num_de_features = num_en_features - 1
        self.task_groups = cfg.TASK_TO_DE_GROUPS
        for group_name, group_cfg in self.task_groups.items():
            if len(group_cfg) == 0:
                continue
            group_stage = group_cfg[-1]
            if group_stage > 0:
                decoder = create_decoder()
                del decoder.blocks[group_stage:]
                self.add_module(group_name + '_decoder', decoder)
            if group_stage < self.num_de_features:
                for task in group_cfg[:-1]:
                    task_decoder = create_decoder()
                    del task_decoder.blocks[:group_stage]
                    task_decoder.num_en_features -= group_stage
                    self.add_module(task + '_decoder', task_decoder)

        self.task_to_encoders = {}
        for task, en_block in self.task_to_en_block.items():
            if 0 < en_block < num_en_features:
                task_encoder = TaskEncoder(
                    en_block, num_en_features, create_encoder, create_basenet)
                self.add_module(task + '_encoder', task_encoder)
                self.task_to_encoders.update({task: task_encoder})
            else:
                self.task_to_encoders.update({task: None})

        self.layer_names = []
        for task in tasks:
            if hasattr(self, f'{task}_decoder'):
                self.layer_names.append(f'{task}_decoder.blocks.0')

    def forward(self, backbone_features):
        hooks, activations = None, None
        if len(self.layer_names) > 0:
            hooks, activations = generic_utils.register_forward_hooks(
                self, self.layer_names)

        uni_en_features = self.uninet_encoder(backbone_features[-1])
        encoder_features = list(backbone_features) + list(uni_en_features)

        task_to_en_feats = {}
        task_to_de_feats = {}
        for task, en_block in self.task_to_en_block.items():
            encoder = self.task_to_encoders[task]
            task_en_feats = encoder_features
            if encoder is not None:
                en_feats = encoder(encoder_features[en_block - 1])
                task_en_feats = encoder_features[:en_block] + en_feats
            task_to_en_feats.update({task: task_en_feats})

        # TODO: assuming all encoder features are the same... group encoders too..
        for group_name, group_cfg in self.task_groups.items():
            if len(group_cfg) == 0:
                continue
            group_stage = group_cfg[-1]
            if group_stage > 0:
                group_de_feats = getattr(self, f'{group_name}_decoder')(
                    encoder_features)
                task_de_feats = group_de_feats
                for task in group_cfg[:-1]:
                    if group_stage < self.num_de_features:
                        task_de_feats += getattr(self, f'{task}_decoder')(
                            encoder_features, de_feats=group_de_feats)
                    task_to_de_feats.update({task: task_de_feats})
            else:
                for task in group_cfg[:-1]:
                    task_de_feats = getattr(self, f'{task}_decoder')(
                        encoder_features)
                    task_to_de_feats.update({task: task_de_feats})

        if hooks is not None:
            for hook in hooks:
                hook.remove()

        return task_to_de_feats, task_to_en_feats, activations


class TaskEncoder(nn.Module):

    def __init__(self, block, num_en_features, create_encoder, create_basenet):
        super(TaskEncoder, self).__init__()
        self.block = block
        self.num_en_features = num_en_features
        uni_en, _ = create_encoder()
        base = create_basenet()
        base = base.backbone

        available_layers = [base.layer1, base.layer2, base.layer3, base.layer4]
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(uni_en, layer):
                available_layers.append(getattr(uni_en, layer))

        for idx in range(block, num_en_features):
            self.add_module(f'layer{idx}', available_layers[idx])

    def forward(self, x):
        outs = [x]
        for idx in range(self.block, self.num_en_features):
            l_fn = getattr(self, f'layer{idx}')
            feats = l_fn(outs[-1])
            outs.append(feats)

        outs.pop(0)
        return outs


class UninetHead(nn.Module):
    def __init__(self, tasks, cfg, norm_layer, **kwargs):

        super(UninetHead, self).__init__()
        task_to_head_fn = {'segment': get_segment_head, 'depth': get_depth_head,
                           'sem_cont': get_sem_cont_head,
                           'sur_nor': get_sur_nor_head,
                           'ae': get_autoencoder_head}

        self.tasks = tasks
        kwargs.update({'norm_layer': norm_layer})
        # -1 as decoder produces one less feature...
        num_en_features = cfg.MODEL.ENCODER.NUM_EN_FEATURES - 1
        for task in tasks:
            head_module = task_to_head_fn[task](cfg, num_en_features, **kwargs)
            self.add_module(task + '_head', head_module)

    def forward(self, task_to_de_feats, task_to_en_feats,
                image_size, targets=None):
        results = OrderedDict()
        for task in self.tasks:
            kwargs = {'encoder_features': task_to_en_feats[task]}
            results[task] = getattr(self, task + '_head')(
                task_to_de_feats[task], image_size, **kwargs)

        return results


def get_uninet_hierarchical(backbone='resnet50', pretrained=True, **kwargs):
    tasks = kwargs.pop('tasks')
    norm_layer = kwargs.pop('norm_layer')
    cfg = kwargs.pop('cfg')
    model = UninetHierarchical(
        backbone, tasks, norm_layer, cfg, pretrained, **kwargs)

    return model
