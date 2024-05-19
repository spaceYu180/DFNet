import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import ChannelChecker
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck
from util.common import ScaleInOutput
from models.backbone.DFNet import *

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))
        self.dl = opt.dual_label
        self.auxiliary_head = False

        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)

        self.check_channels = ChannelChecker(self.backbone, self.inplanes, opt.input_size)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)
        self._model_summary(opt.input_size)

    def forward(self, xa, xb, tta=False):
        if not tta:
            return self.forward_once(xa, xb)
        else:
            return self.forward_tta(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backbone(xa)      
        fa1, fa2, fa3, fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, fb4 = self.backbone(xb)
        fb1, fb2, fb3, fb4 = self.check_channels(fb1, fb2, fb3, fb4)

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4

        change, _ = self.neck(ms_feats)

        out = self.head_forward(ms_feats, change, out_size=(h_input, w_input))

        return out

    def forward_tta(self, xa, xb):
        bs, c, h, w = xa.shape
        mutil_scales = [1.0, 0.834, 0.667, 0.542]

        out1, out2 = 0, 0
        for single_scale in mutil_scales:
            single_scale = (int((h * single_scale) / 32) * 32, int((w * single_scale) / 32) * 32)
            xa_size = F.interpolate(xa, single_scale, mode='bilinear', align_corners=True)
            xb_size = F.interpolate(xb, single_scale, mode='bilinear', align_corners=True)

            out_1 = self.forward_once(xa_size, xb_size)

            if self.dl:
                out1_1, out1_2 = out_1[0], out_1[1]

            else:
                out1_1 = out_1

                out1 += F.interpolate(out1_1,
                                      size=(h, w), mode='bilinear', align_corners=True)

        return (out1, out2) if self.dl else out1

    def head_forward(self, ms_feats, change, out_size):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats

        out1 = F.interpolate(self.head1(change), size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(change), size=out_size,
                             mode='bilinear', align_corners=True) if self.dl else None

        if self.training and self.auxiliary_head:
            aux_stage1_out1 = F.interpolate(self.aux_stage1_head1(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage1_out2 = F.interpolate(self.aux_stage1_head2(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage2_out1 = F.interpolate(self.aux_stage2_head1(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage2_out2 = F.interpolate(self.aux_stage2_head2(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage3_out1 = F.interpolate(self.aux_stage3_head1(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage3_out2 = F.interpolate(self.aux_stage3_head2(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage4_out1 = F.interpolate(self.aux_stage4_head1(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage4_out2 = F.interpolate(self.aux_stage4_head2(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            return (out1, out2,
                    aux_stage1_out1, aux_stage1_out2, aux_stage2_out1, aux_stage2_out2,
                    aux_stage3_out1, aux_stage3_out2, aux_stage4_out1, aux_stage4_out2) \
                if self.dl else (out1, aux_stage1_out1, aux_stage2_out1, aux_stage3_out1, aux_stage4_out1)
        else:
            return (out1, out2) if self.dl else out1

    def _init_weight(self, pretrain=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _model_summary(self, input_size):
        input_sample = torch.randn(1, 3, input_size, input_size)

    def _create_backbone(self, backbone):
        if 'dfnet' in backbone:
            self.backbone = dfnet_large()
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)

    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None

        if self.auxiliary_head:
            self.aux_stage1_head1 = FCNHead(self.inplanes * 2, 2)
            self.aux_stage1_head2 = FCNHead(self.inplanes * 2, 2) if self.dl else None

            self.aux_stage2_head1 = FCNHead(self.inplanes * 4, 2)
            self.aux_stage2_head2 = FCNHead(self.inplanes * 4, 2) if self.dl else None

            self.aux_stage3_head1 = FCNHead(self.inplanes * 8, 2)
            self.aux_stage3_head2 = FCNHead(self.inplanes * 8, 2) if self.dl else None

            self.aux_stage4_head1 = FCNHead(self.inplanes * 16, 2)
            self.aux_stage4_head2 = FCNHead(self.inplanes * 16, 2) if self.dl else None


class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-"*50+"\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1, cd_pred2 = None, None

        for i, model in enumerate(self.models_list):
            outs = model(xa, xb, tta)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1)
                _, cd_pred2 = torch.max(out2, 1)
            elif self.method == "vote":
                _, out1_tmp = torch.max(outs[0], 1)
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / i >= 0.5
                cd_pred2 = out2 / i >= 0.5

        if self.models_list[0].dl:
            return cd_pred1, cd_pred2
        else:
            return cd_pred1


class ModelEMA:

    def __init__(self, model, decay=0.96):
        self.shadow1 = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.decay = decay
        for p in self.shadow1.parameters():
            p.requires_grad_(False)

        self.shadow2 = deepcopy(self.shadow1)
        self.shadow3 = deepcopy(self.shadow1)
        self.update_count = 0

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.shadow1.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1. - self.decay) * msd[k].detach()
            for k, v in self.shadow2.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.95
                    v += (1. - 0.95) * msd[k].detach()
            for k, v in self.shadow3.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.94
                    v += (1. - 0.94) * msd[k].detach()
        self.update_count += 1

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelSWA:

    def __init__(self, total_epoch=300):
        self.update_count = 0
        self.epoch_threshold = int(total_epoch * 0.8)
        self.swa_model = None

    def update(self, model):
        if self.update_count >= self.epoch_threshold:
            with torch.no_grad():
                if self.swa_model is None:
                    self.swa_model = deepcopy(model.module) if self.is_parallel(model) else deepcopy(model)
                else:
                    msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
                    for k, v in self.swa_model.state_dict().items():
                        if v.dtype.is_floating_point:
                            v *= (self.update_count - self.epoch_threshold)
                            v += msd[k].detach()
                            v /= (self.update_count - self.epoch_threshold + 1)
        self.update_count += 1

    def save(self, swa_ckp_dir_path):
        if self.update_count >= self.epoch_threshold:
            swa_file_path = os.path.join(swa_ckp_dir_path, "swa_{}_{}.pt".format(
                self.update_count-1, self.update_count-self.epoch_threshold))
            torch.save(self.swa_model, swa_file_path)

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
