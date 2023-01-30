# Author: Zylo117

import math

import torch
from torch import nn
import torch.nn.functional as F

from .efficientdet import BiFPN, Regressor, Classifier, EfficientNet, SeparableConvBlock
# from efficientdet.utils import Anchors

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = num_layers
        self.heads = heads

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = 512#[512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        self.backbone_net = EfficientNet(self.backbone_compound_coef[self.compound_coef])

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True if self.compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])


        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    SeparableConvBlock(320, head_conv, 
                        norm=False, activation=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, 
                        kernel_size=1, stride=1, 
                        padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = SeparableConvBlock(320, head_conv, 
                        norm=False, activation=False)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        p1, p2, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)
        o1, o2, o3, o4, o5 = features
        _o1 = F.interpolate(o1, (128, 128), mode='bilinear', align_corners=True)
        _o2 = F.interpolate(o2, (128, 128), mode='bilinear', align_corners=True)
        _o3 = F.interpolate(o3, (128, 128), mode='bilinear', align_corners=True)
        _o4 = F.interpolate(o4, (128, 128), mode='bilinear', align_corners=True)
        _o5 = F.interpolate(o5, (128, 128), mode='bilinear', align_corners=True)
        x = torch.cat((_o1, _o2, _o3, _o4, _o5), dim=1)
        # regression = self.regressor(features)
        # classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        # return features, regression, classification, anchors
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_backbone(self):
        state_dict = torch.load("/home/habibi/CenterNet/models/efficientdet-d0.pth")
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print("LOAD PRE-TRAINED")
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

def get_efficient_net(num_layers, heads, head_conv):
  #block_class, layers = resnet_spec[num_layers]

  model = EfficientDetBackbone(num_layers, heads, head_conv=head_conv)
  #model.init_backbone()
  return model