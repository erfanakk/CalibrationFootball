# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

#keypoint_hrnet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.hub import load_state_dict_from_url


from dataclasses import dataclass
from typing import List, Optional

IMAGENET_URL = "https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ"


@dataclass
class StageConfig:
    NUM_MODULES: int
    NUM_BRANCHES: int
    BLOCK: str
    NUM_BLOCKS: List[int]
    NUM_CHANNELS: List[int]
    FUSE_METHOD: str

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class ExtraConfig:
    FINAL_CONV_KERNEL: int
    STAGE2: StageConfig
    STAGE3: StageConfig
    STAGE4: StageConfig

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class ModelConfig:
    NAME: str
    TARGET_TYPE: str
    SIGMA: int
    EXTRA: ExtraConfig

    def __getitem__(self, key):
        return getattr(self, key)


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class KeypointHRNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.EXTRA
        super(KeypointHRNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg["EXTRA"]["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg["EXTRA"]["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg["EXTRA"]["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)

        self.out_channels = pre_stage_channels[0]

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg.NUM_BRANCHES:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg.NUM_BRANCHES:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list[0]

    def init_weights(self):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)

    def get_n_channels_out(self):
        return self.out_channels


def get_hrnet_model(pretrained=False):
    extra = ExtraConfig(
        FINAL_CONV_KERNEL=1,
        STAGE2=StageConfig(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK="BASIC",
            NUM_BLOCKS=[4, 4],
            NUM_CHANNELS=[48, 96],
            FUSE_METHOD="SUM",
        ),
        STAGE3=StageConfig(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK="BASIC",
            NUM_BLOCKS=[4, 4, 4],
            NUM_CHANNELS=[48, 96, 192],
            FUSE_METHOD="SUM",
        ),
        STAGE4=StageConfig(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK="BASIC",
            NUM_BLOCKS=[4, 4, 4, 4],
            NUM_CHANNELS=[48, 96, 192, 384],
            FUSE_METHOD="SUM",
        ),
    )

    model_config = ModelConfig(
        NAME="hrnet",
        TARGET_TYPE="gaussian",
        SIGMA=3,
        EXTRA=extra,
    )

    model = KeypointHRNet(model_config)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(IMAGENET_URL, model_dir=".", file_name="model.pth"), strict=False)
    else:
        model.init_weights()
    return model


class KeypointModel(nn.Module):
    def __init__(self, n_heatmaps: int = 57):
        super(KeypointModel, self).__init__()
        backbone = get_hrnet_model(False)
        head = nn.Conv2d(
            in_channels=backbone.get_n_channels_out(),
            out_channels=n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )
        self.unnormalized_model = nn.Sequential(
            backbone,
            head,
        )
        self.n_heatmaps = n_heatmaps

    def forward(self, x: torch.Tensor):
        """
        x shape must be of shape (N,3,H,W)
        returns tensor with shape (N, n_heatmaps, H,W)
        """
        return torch.sigmoid(self.forward_unnormalized(x))

    def forward_unnormalized(self, x: torch.Tensor):
        return self.unnormalized_model(x)

    # def get_keypoints(self, x: torch.Tensor):
    #     """
    #     x shape must be of shape (N,3,H,W)
    #     returns list of list of tuples
    #     Coordinates are in normalized coordinates
    #     Optimized for batch processing with GPU operations
    #     """
    #     batch_heatmaps = self.forward(x)  # Shape: (N, n_heatmaps, H, W)
    #     batch_size, n_heatmaps, height, width = batch_heatmaps.shape
        
    #     # Reshape to (N * n_heatmaps, H, W) for vectorized operations
    #     heatmaps_flat = batch_heatmaps.view(-1, height, width)
        
    #     # Find max values and indices for all heatmaps at once (GPU operation)
    #     max_vals, max_indices = torch.max(heatmaps_flat.view(heatmaps_flat.size(0), -1), dim=1)
        
    #     # Convert linear indices to 2D coordinates
    #     y_coords = max_indices // width
    #     x_coords = max_indices % width
        
    #     # Reshape back to (N, n_heatmaps)
    #     max_vals = max_vals.view(batch_size, n_heatmaps)
    #     y_coords = y_coords.view(batch_size, n_heatmaps)
    #     x_coords = x_coords.view(batch_size, n_heatmaps)
        
    #     # Normalize coordinates
    #     x_norm = x_coords.float() / 240.0
    #     y_norm = y_coords.float() / 135.0
        
    #     # Create mask for valid keypoints (score > 0.01)
    #     valid_mask = max_vals > 0.01
        
    #     # Convert to list format
    #     batch_coords = []
    #     for b in range(batch_size):
    #         coords = []
    #         for i in range(n_heatmaps):
    #             if valid_mask[b, i]:
    #                 coords.append((i, x_norm[b, i].item(), y_norm[b, i].item(), max_vals[b, i].item()))
    #         batch_coords.append(coords)
        
    #     return batch_coords

    def get_keypoints(self, x: torch.Tensor):
        batch_heatmaps = self.forward(x)
        batch_size, n_heatmaps, height, width = batch_heatmaps.shape
        
        heatmaps_flat = batch_heatmaps.view(-1, height, width)
        max_vals, max_indices = torch.max(heatmaps_flat.view(heatmaps_flat.size(0), -1), dim=1)
        
        y_coords = max_indices // width
        x_coords = max_indices % width
        max_vals = max_vals.view(batch_size, n_heatmaps)
        y_coords = y_coords.view(batch_size, n_heatmaps)
        x_coords = x_coords.view(batch_size, n_heatmaps)
        
        x_norm = x_coords.float() / 240.0
        y_norm = y_coords.float() / 135.0
        valid_mask = max_vals > 0.01
        
        # Transfer to CPU only once
        x_norm = x_norm.cpu().detach().numpy()
        y_norm = y_norm.cpu().detach().numpy()
        max_vals = max_vals.cpu().detach().numpy()
        valid_mask = valid_mask.cpu().detach().numpy()
        
        batch_coords = []
        for b in range(batch_size):
            coords = []
            for i in range(n_heatmaps):
                if valid_mask[b, i]:
                    coords.append((i, x_norm[b, i], y_norm[b, i], max_vals[b, i]))
            batch_coords.append(coords)
        
        return batch_coords