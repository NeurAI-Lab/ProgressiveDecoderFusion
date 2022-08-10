import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.common import conv1x1, conv3x3
from encoding_custom.backbones.resnet import Bottleneck


class AEDecoder(nn.Module):
    def __init__(self, cfg, num_features, **kwargs):
        super(AEDecoder, self).__init__()
        en_feat_channels = kwargs['en_feat_channels']
        self.blocks = nn.ModuleList()
        # -2 would upsample only upto 1/16 input size..
        for i in range(len(en_feat_channels) - 2):
            self.blocks.append(Bottleneck(
                en_feat_channels[-1], en_feat_channels[-1] // Bottleneck.expansion))

    def forward(self, encoder_features):
        out = encoder_features[-1]
        for block in self.blocks:
            out = block(out)
            out = F.interpolate(out, [val * 2 for val in out.size()[-2:]],
                                mode="bilinear", align_corners=False)

        return out


class AutoEncoder(nn.Module):
    def __init__(self, cfg, num_features, **kwargs):
        super(AutoEncoder, self).__init__()

        en_feat_channels = kwargs['en_feat_channels']
        self.num_features = num_features

        self.ae_decoder = AEDecoder(cfg, num_features, **kwargs)
        self.final_conv = conv3x3(en_feat_channels[-1], 3, bias=True)

    def forward(self, decoder_features, mask_size, **kwargs):
        # not using the decoder features...
        # this class can be later modified to accommodate more complex
        # AE structures...
        encoder_features = kwargs['encoder_features']
        decoded_feats = self.ae_decoder(encoder_features)
        out = self.final_conv(decoded_feats)
        out = F.interpolate(out, size=mask_size, mode='bilinear',
                            align_corners=False)
        return {'reconst': out, 'en_feats': encoder_features}


class AutoEncoderHead(nn.Module):
    def __init__(self, cfg, num_features, **kwargs):
        super(AutoEncoderHead, self).__init__()

        in_planes = cfg.MODEL.AE.INPLANES
        out_planes = cfg.MODEL.AE.OUTPLANES
        self.num_features = num_features

        self.convs = nn.ModuleList()
        for i in range(self.num_features):
            self.convs.append(conv1x1(in_planes * 4, out_planes, bias=False))
        self.final_conv = conv3x3(in_planes * self.num_features, 3, bias=True)

    def forward(self, decoder_features, mask_size, **kwargs):
        outs = []
        for i in range(self.num_features):
            x = self.convs[i](decoder_features[i])
            outs.append(
                F.interpolate(x, (mask_size[0] // 4, mask_size[1] // 4),
                              mode="bilinear", align_corners=False))
        all_outs = torch.cat(outs, dim=1)
        out = self.final_conv(all_outs)

        out = F.interpolate(out, size=mask_size, mode='bilinear',
                            align_corners=False)
        return {'reconst': out, 'en_feats': kwargs['encoder_features']}
