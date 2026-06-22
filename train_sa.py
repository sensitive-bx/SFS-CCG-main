import argparse
import time
import datetime
import os

import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端，不启动 GUI
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from losses import *
from models.model_zoo import get_segmentation_model

from utils.sagan import Discriminator
from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from utils.flops import cal_multi_adds, cal_param_size

from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet

import logging
import random


def set_seed(seed):
    logging.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')  # old : [512, 1024]
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-cwd-logit", type=float, default=0., help="lambda cwd logit")
    parser.add_argument("--lambda_cam", type=float, default=0., help="lambda_cam")
    parser.add_argument("--lambda_campam", type=float, default=0., help="lambda_cam")
    parser.add_argument("--lambda_pam", type=float, default=0., help="lambda_pam")
    parser.add_argument("--cwd_temperature", type=float, default=1.0, help="normalize temperature")
    parser.add_argument("--sa_temperature", type=float, default=1.0, help="normalize temperature SA")

    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='2')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    parser.add_argument("--lambda_stca", type=float, default=1., help="lambda for STCA_ASCM_M_CKA")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ 新增参数以支持差异化蒸馏
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--lambda-edd", type=float, default=1.0, help="lambda for error-driven distillation loss")
    parser.add_argument("--ohem_thresh", type=float, default=0.7, help="OHEM threshold for hard pixel mining")
    parser.add_argument('--EDD_method', type=str, default='XOR', choices=['XOR', 'OR', 'AND', 'OHEM'],
                        help='the method EDD use to define difficult regions')
    parser.add_argument('--warmup_iter', type=int, default=8000,
                        help='per iters to val')
    parser.add_argument('--use_weight', action='store_true', default=True)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 新增参数，用于动态控制 S->T 注意力
    parser.add_argument('--s_to_t_threshold', type=float, default=0.1,
                        help='Hard pixel ratio threshold to activate S->T attention in STCA. (default: 0.1)')

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 从 train_onlykd_two_stage_diff.py 引入 OhemCrossEntropy2d 类
# +++ 这个类实现了基于在线硬样本挖掘（OHEM）的交叉熵损失。
# +++ 它的核心功能是识别出模型最难分类的像素（即预测置信度最低的像素），
# +++ 并只在这些“硬”像素上计算损失，从而引导模型关注困难样本。
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, args=None, reduction='mean', alignment_thresh=0.4, momentum=0.9,
                 **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        logger.info(f'OHEM thresh is {thresh}')

        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.reduction = reduction

        # === 新增参数 ===
        self.alignment_thresh = alignment_thresh  # 触发阈值，推荐 0.4~0.5
        self.momentum = momentum  # 动量系数，越接近1越平滑

        # 运行时状态
        self.register_buffer('running_iou', torch.tensor(0.0))  # 全局滑动平均IoU
        self.s_to_t_active_latch = False  # 门锁：一旦为True，永久为True


        # if use_weight:
        #     # 城市景观数据集的类别权重，用于处理类别不平衡问
        #     weight = torch.FloatTensor(
        #         [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
        #          1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        #          1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        #     self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
        #                                                reduction=self.reduction)
        # else:
        #     self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=self.reduction)

        # 支持按数据集自动选择类别权重：citys (19 类) / voc (21 类)，或在未知数据集时回退为均匀权重/无权重
        if use_weight:
            weight = None
            if args is not None and getattr(args, 'dataset', '') == 'citys':
                # 城市景观数据集的类别权重，用于处理类别不平衡问题
                weight = torch.FloatTensor(
                    [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                     1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                     1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            elif args is not None and getattr(args, 'dataset', '') == 'voc':
                # VOC (Pascal VOC) 通常包含 21 个类别（包括背景），这里使用均匀权重作为默认。
                # 如果你有 VOC 的类频率统计，可以替换为更合适的权重向量。
                weight = torch.FloatTensor([1.0] * 21)
            else:
                # fallback: 若未识别数据集或不提供权重信息，使用无权重（等价于 weight=None）
                weight = None

            if weight is not None:
                self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                                           reduction=self.reduction)
            else:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=self.reduction)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=self.reduction)

        self.args = args

    def calc_mask_iou(self, mask_s, mask_t):
        """计算两个掩码的 Intersection over Union"""
        # 展平
        mask_s = mask_s.view(-1)
        mask_t = mask_t.view(-1)

        intersection = (mask_s & mask_t).sum().float()
        union = (mask_s | mask_t).sum().float()

        # 防止除以0 (如果两张图都没有Hard Sample，认为是一致的，返回1.0)
        if union == 0:
            return torch.tensor(1.0, device=mask_s.device)

        return intersection / union

    def get_mask(self, pred, target):
        """
        根据预测和目标生成OHEM掩码。
        :param pred: 模型的预测 logits, [N, C, H, W]
        :param target: 真实标签, [N*H*W]
        :return: valid_mask (考虑了ignore_index和OHEM的最终掩码), kept_mask (仅由OHEM生成的掩码)
        """
        n, c, h, w = pred.size()
        # 忽略标签为 ignore_index 的像素
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)


        if self.min_kept > num_valid:
            print("Labels: {}".format(num_valid))
        elif num_valid > 0:
            # 获取每个像素对应正确类别的预测概率
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh
            # 如果设置了 min_kept，确保至少保留 min_kept 个最难的样本
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                # 如果最难的第 min_kept 个样本的概率比预设的 thresh 还高，那么就用这个概率作为新的阈值
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]

            # 概率小于等于阈值的像素被认为是“硬”样本，需要保留
            kept_mask = mask_prob.le(threshold)
            # 最终的掩码是初始有效掩码和OHEM掩码的交集
            valid_mask = valid_mask * kept_mask

        return valid_mask, kept_mask

    # def forward_with_teacher_student(self, pred_student, pred_teacher, target):
    #     """
    #     计算师生模型之间的差异化蒸馏损失。
    #     这个函数是实现EDD（Error-Driven Distillation）的核心。
    #     """
    #     B, H, W = target.size()
    #     # 确保师生预测图尺寸一致
    #     pred_teacher = F.interpolate(pred_teacher, (H, W), mode='bilinear', align_corners=True)
    #     pred_student = F.interpolate(pred_student, (H, W), mode='bilinear', align_corners=True)
    #
    #     n, c, h, w = pred_teacher.size()
    #     target = target.view(-1)
    #
    #     # 分别获取学生和教师模型的OHEM掩码
    #     valid_mask_student, kept_mask_student = self.get_mask(pred_student.clone(), target.clone())
    #     valid_mask_teacher, kept_mask_teacher = self.get_mask(pred_teacher.clone(), target.clone())
    #
    #     # 根据设定的方法（XOR, OR, AND等）计算师生之间的“差异区域”
    #     if self.args.EDD_method == 'XOR':
    #         # 异或：只保留师生模型一个认为是硬样本，另一个认为是简单样本的区域
    #         mask_diff = valid_mask_student ^ valid_mask_teacher
    #     elif self.args.EDD_method == 'OR':
    #         # 或：保留任何一方认为是硬样本的区域
    #         mask_diff = valid_mask_student | valid_mask_teacher
    #     elif self.args.EDD_method == 'AND':
    #         # 与：只保留双方都认为是硬样本的区域
    #         mask_diff = valid_mask_student & valid_mask_teacher
    #     elif self.args.EDD_method == 'OHEM':
    #         # 只使用学生模型的OHEM结果
    #         mask_diff = valid_mask_student
    #
    #     # keep_mask_union 是教师和学生OHEM掩码的并集
    #     keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)
    #
    #     # 首先，只在教师或学生认为是硬样本的区域（keep_mask_union）内保留标签
    #     target = target * keep_mask_union.long()
    #     # 然后，在这些区域中，只在“差异区域”（mask_diff）内计算损失，其他区域的标签设为ignore_index
    #     target = target.masked_fill_(~mask_diff, self.ignore_index)
    #     target = target.view(n, h, w)
    #
    #     # 在最终确定的困难且有差异的区域上，计算学生模型的交叉熵损失
    #     return self.criterion(pred_student, target), mask_diff.view(n, h, w)

    def forward_with_teacher_student(self, pred_student, pred_teacher, target):
        """
        计算师生模型之间的差异化蒸馏损失。
        这个函数是实现EDD（Error-Driven Distillation）的核心。
        """
        B, H, W = target.size()
        # 确保师生预测图尺寸一致
        pred_teacher = F.interpolate(pred_teacher, (H, W), mode='bilinear', align_corners=True)
        pred_student = F.interpolate(pred_student, (H, W), mode='bilinear', align_corners=True)

        n, c, h, w = pred_teacher.size()
        target = target.view(-1)

        # 分别获取学生和教师模型的OHEM掩码
        # valid_mask_student 包含了 ignore_index 的处理逻辑，kept_mask 是纯粹的 OHEM 结果
        valid_mask_student, kept_mask_student = self.get_mask(pred_student.clone(), target.clone())
        valid_mask_teacher, kept_mask_teacher = self.get_mask(pred_teacher.clone(), target.clone())

        # === [核心改进] 计算认知对齐度 ===
        with torch.no_grad():
            # 计算当前Batch的IoU
            current_iou = self.calc_mask_iou(kept_mask_student, kept_mask_teacher)

            # 更新滑动平均 (Running Average)
            # 如果是第一次运行(为0)，直接赋值
            if self.running_iou == 0:
                self.running_iou = current_iou
            else:
                self.running_iou = self.momentum * self.running_iou + (1 - self.momentum) * current_iou

            # 判断是否激活 S->T (使用 Latch 机制)
            if not self.s_to_t_active_latch:
                if self.running_iou > self.alignment_thresh:
                    self.s_to_t_active_latch = True
                    logger.info(f"Feature Alignment Reached! (IoU: {self.running_iou:.4f}). Activating S->T Attention.")

        # 根据设定的方法（XOR, OR, AND等）计算师生之间的“差异区域”
        if self.args.EDD_method == 'XOR':
            # 异或：只保留师生模型一个认为是硬样本，另一个认为是简单样本的区域
            mask_diff = valid_mask_student ^ valid_mask_teacher
        elif self.args.EDD_method == 'OR':
            # 或：保留任何一方认为是硬样本的区域
            mask_diff = valid_mask_student | valid_mask_teacher
        elif self.args.EDD_method == 'AND':
            # 与：只保留双方都认为是硬样本的区域
            mask_diff = valid_mask_student & valid_mask_teacher
        elif self.args.EDD_method == 'OHEM':
            # 只使用学生模型的OHEM结果
            mask_diff = valid_mask_student

        # keep_mask_union 是教师和学生OHEM掩码的并集
        keep_mask_union = torch.logical_or(kept_mask_student, kept_mask_teacher)

        # 首先，只在教师或学生认为是硬样本的区域（keep_mask_union）内保留标签
        target = target * keep_mask_union.long()
        # 然后，在这些区域中，只在“差异区域”（mask_diff）内计算损失，其他区域的标签设为ignore_index
        target = target.masked_fill_(~mask_diff, self.ignore_index)
        target = target.view(n, h, w)

        # 在最终确定的困难且有差异的区域上，计算学生模型的交叉熵损失
        return self.criterion(pred_student, target), mask_diff.view(n, h, w), self.s_to_t_active_latch

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data,
                                          list_path='./dataset/list/cityscapes/train.lst',
                                          max_iters=args.max_iterations * args.batch_size,
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data,
                                        list_path='./dataset/list/cityscapes/val.lst',
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt',
                                            max_iters=args.max_iterations * args.batch_size,
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt',
                                           max_iters=args.max_iterations * args.batch_size,
                                           ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True,
                                           mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_test_list.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations * args.batch_size,
                                        ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt',
                                                  max_iters=args.max_iterations * args.batch_size,
                                                  ignore_label=args.ignore_label,
                                                  crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model,
                                              backbone=args.teacher_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base='None',
                                              pretrained=args.teacher_pretrained,
                                              aux=True,
                                              norm_layer=nn.BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model,
                                              backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base,
                                              pretrained='None',
                                              aux=args.aux,
                                              norm_layer=BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.device)

        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        self.D_model = Discriminator(preprocess_GAN_mode=1, input_channel=train_dataset.num_class,
                                     distributed=args.distributed).cuda()

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        x = torch.randn(1, 3, 512, 512).cuda()
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_cwd = CriterionCWD(s_channels, t_channels, norm_type='channel', divergence='kl',
                                          temperature=args.cwd_temperature).to(self.device)

        #self.criterion_sa = CriterionSA(args.cwd_temperature, args.sa_temperature).cuda()

        self.criterion_sa = CriterionSA(cwd_temperature=args.cwd_temperature,
                                        sa_temperature=args.sa_temperature,
                                        num_masks=4,
                                        hidden_dim=64,
                                        align_method='cka',
                                        diversity_weight=0.05,
                                        max_spatial_size=48,
                                        lambda_freq=0.5).to(self.device)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++ 实例化 OhemCrossEntropy2d 损失函数
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.OHEMcriterion = OhemCrossEntropy2d(ignore_index=args.ignore_label, thresh=args.ohem_thresh, use_weight=args.use_weight, args=args).to(
            self.device)
        print(args.use_weight)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        params_list.append(self.criterion_cwd)
        params_list.append(self.criterion_sa)

        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                   self.D_model.parameters()),
                                            4e-4, [0.9, 0.99])

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model,
                                                               device_ids=[args.local_rank],
                                                               output_device=args.local_rank)
            self.criterion_cwd = nn.parallel.DistributedDataParallel(self.criterion_cwd,
                                                                     device_ids=[args.local_rank],
                                                                     output_device=args.local_rank)
            # self.criterion_sa = nn.parallel.DistributedDataParallel(self.criterion_sa,
            #                                                     device_ids=[args.local_rank],
            #                                                     output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        self.best_pixAcc = 0.0

        self.best_mDice = 0.0  # <--- 新增

        #用于记录S->T是否激活的状态变量
        self.s_to_t_activated_logged = False

        self.s_to_t_threshold = args.s_to_t_threshold

        # [新增] 用于存储自动抓取的典型样本
        self.sentinels = {}

        self.global_max_var = -1.0
        self.global_min_var = 999999.0


    # ------------------------------------------------------------------
    # [新增] 辅助函数：绘制 Mask + Filter + Stats 的组合图
    # ------------------------------------------------------------------
    def visualize_sentinel_snapshot(self, iteration, name, img_tensor, mask_tensor, filter_tensor, sigma, gamma,
                                    freq_loss):
        """
        生成一张组合图：
        左图: 原始图像 + Mask 热力图叠加 (展示空间关注点)
        右图: 频域滤波器 W (展示频域关注点)
        标题: 包含 Sigma, Gamma, FreqLoss 数值
        """
        # 1. 图像处理
        img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        # 2. Mask 处理
        mask = F.interpolate(mask_tensor, size=img_tensor.shape[2:], mode='bilinear', align_corners=False)[0, 0]
        mask = mask.cpu().numpy()

        # =============================================================
        # [核心修改] 3. 重新生成中心化的 Filter 图 (用于可视化)
        # =============================================================
        # 我们不直接画 filter_tensor (左上角原点)，而是用 sigma/gamma
        # 在一个中心化网格上重新计算一遍 W，这样低频就在正中心了。

        # 创建一个 256x256 的网格，坐标范围从 -1 到 1，中心是 (0,0)
        vis_size = 256
        y = np.linspace(-1, 1, vis_size)
        x = np.linspace(-1, 1, vis_size)
        xx, yy = np.meshgrid(x, y)

        # 计算径向距离 r (中心为0，四周为1.414)
        # 我们截断到 1.0，模拟单位圆内的频率响应
        r = np.sqrt(xx ** 2 + yy ** 2)

        # 套用你 sa.py 里的物理公式
        # w_low = exp(-r^2 / 2sigma^2)
        w_low = np.exp(-(r ** 2) / (2 * sigma ** 2))
        w_high = 1.0 - w_low

        # 生成最终的 W 图
        W_centered = gamma * w_high + (1.0 - gamma) * w_low

        # =============================================================

        # 4. 绘图
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 左：Mask
        axs[0].imshow(img)
        axs[0].imshow(mask, cmap='jet', alpha=0.5)
        axs[0].set_title(f"{name}\nSpatial Attention (Mask 0)")
        axs[0].axis('off')

        # 右：中心化 Filter
        # extent 参数确保坐标轴显示正确，vmin/vmax 确保颜色映射一致
        im = axs[1].imshow(W_centered, cmap='plasma', vmin=0, vmax=1, extent=[-1, 1, -1, 1])

        # 加上漂亮的标题和圈注
        axs[1].set_title(
            f"Freq Filter ($W$)\nCenter=LowFreq, Edge=HighFreq\n$\sigma$={sigma:.2f}, $\gamma$={gamma:.2f}")
        axs[1].axis('off')

        # 添加色卡
        cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.set_label('Filter Weight', rotation=270, labelpad=15)

        # 保存
        save_dir = os.path.join(self.args.log_dir, 'vis_proof', name)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'iter_{iteration}.png'), bbox_inches='tight', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # [新增] 探测函数：定期被调用，生成数据和图片
    # ------------------------------------------------------------------
    def probe_training_sentinels(self, iteration):
        if not self.sentinels:
            return

        # 1. 切换到 eval 模式
        # 这对于 BatchNorm 至关重要，否则 Batch=1 时会报错
        self.s_model.eval()
        self.criterion_sa.eval()  # <--- [修改] 必须加上这行！

        # CSV 文件路径
        csv_path = os.path.join(self.args.log_dir, 'vis_proof', 'evolution.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("iteration,type,sigma,gamma,freq_loss,feat_mean,feat_std\n") # <--- 新增

        with torch.no_grad():
            for s_type, (img, tgt) in self.sentinels.items():
                t_out = self.t_model(img)
                s_out = self.s_model(img)

                # 调用 visualization_mode=True
                masks, W_map, sigma, gamma, f_loss, f_mean, f_std = self.criterion_sa(
                    feat_S=s_out[-1],
                    feat_T=t_out[-1],
                    feature_transform="STCA_ASCM_M_CKA",
                    activate_s_to_t=True,
                    visualization_mode=True
                )

                # 获取数值
                s_val = sigma.mean().item()
                g_val = gamma.mean().item()
                l_val = f_loss.item()
                m_val = f_mean.mean().item()
                std_val = f_std.mean().item()

                # 1. 写入 CSV
                with open(csv_path, "a") as f:
                    f.write(f"{iteration},{s_type},{s_val:.4f},{g_val:.4f},{l_val:.4f},{m_val:.4f},{std_val:.4f}\n")

                # 2. 画图
                self.visualize_sentinel_snapshot(
                    iteration, s_type, img, masks, W_map, s_val, g_val, l_val
                )

        # 2. 恢复训练模式
        self.s_model.train()
        self.criterion_sa.train()  # <--- [修改] 必须加上这行，恢复训练状态！

    def decode_segmap(self, label_mask):
        """ 将类别索引转换为彩色图 (Cityscapes 官方配色) """
        if args.dataset == 'citys':
            colors = [
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]
            ]
        elif args.dataset == 'camvid':
            # CamVid 官方定义的 RGB 颜色
            # 0:Sky, 1:Building, 2:Pole, 3:Road, 4:Pavement, 5:Tree, 6:SignSymbol,
            # 7:Fence, 8:Car, 9:Pedestrian, 10:Bicyclist, 11:Unlabelled
            colors = [
                [128, 128, 128],  # 0: Sky (天蓝色/灰色)
                [128, 0, 0],  # 1: Building (深红色)
                [192, 192, 128],  # 2: Pole (米黄色)
                [128, 64, 128],  # 3: Road (紫色)
                [60, 40, 222],  # 4: Pavement (深蓝色)
                [128, 128, 0],  # 5: Tree (橄榄绿)
                [192, 128, 128],  # 6: SignSymbol (粉红色)
                [64, 64, 128],  # 7: Fence (蓝紫色)
                [64, 0, 128],  # 8: Car (深紫色)
                [64, 64, 0],  # 9: Pedestrian (深黄色)
                [0, 128, 192],  # 10: Bicyclist (青蓝色)
                [0, 0, 0]  # 11: Void (黑色)
            ]
        else:
            # 兜底方案：随机颜色
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(256, 3)).tolist()


        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(colors):
            rgb[label_mask == label] = color
        return rgb

    # def save_paper_vis(self, img_tensor, target_tensor, masks_tensor, iteration, save_dir, name="camvid_sample"):
    #     """
    #     生成 CamVid 论文专用对比图：Input | GT | M1 | M2 | M3 | M4
    #     """
    #     # 1. 基础图像处理
    #     img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    #     # 反归一化到 0-1
    #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #     H, W = img.shape[:2]
    #
    #     # 2. 生成彩色 Groundtruth
    #     gt_label = target_tensor[0].cpu().numpy()
    #     gt_color = self.decode_segmap_camvid(gt_label)
    #
    #     # 3. 掩码插值
    #     masks = F.interpolate(masks_tensor, size=(H, W), mode='bilinear', align_corners=False)[0]
    #     masks = masks.cpu().detach().numpy()  # [4, H, W]
    #
    #     # 4. 创建 1行6列 画布 (Input + GT + 4个Mask)
    #     fig, axs = plt.subplots(1, 6, figsize=(30, 5), facecolor='white')
    #     plt.subplots_adjust(wspace=0.02)  # 极小间距，更美观
    #
    #     # (a) Input Image
    #     axs[0].imshow(img)
    #     axs[0].set_title("Input Image", fontsize=14)
    #
    #     # (b) Groundtruth
    #     axs[1].imshow(gt_color)
    #     axs[1].set_title("Groundtruth (GT)", fontsize=14)
    #
    #     # (c)-(f) 绘制 4 个 Mask
    #     titles = ["Mask 1: Edges", "Mask 2: Textures", "Mask 3: Interiors", "Mask 4: Background"]
    #
    #     for i in range(4):
    #         ax = axs[i + 2]
    #         mask = masks[i]
    #
    #         # === 核心修改：归一化 ===
    #         # 即使数值很接近，归一化也能强行拉开差距，让热力图出现红色和蓝色
    #         m_min, m_max = mask.min(), mask.max()
    #         if m_max - m_min > 1e-5:
    #             mask = (mask - m_min) / (m_max - m_min)
    #
    #         # 应用 Jet 配色
    #         m_color = cm.jet(mask)[:, :, :3]
    #
    #         # 叠加显示
    #         ax.imshow(img)
    #         ax.imshow(m_color, alpha=0.5)  # 透明度 0.5 效果最好
    #         ax.set_title(titles[i], fontsize=14)
    #
    #     # 统一移除坐标轴
    #     for ax in axs:
    #         ax.axis('off')
    #
    #     # 5. 存储
    #     save_path = os.path.join(save_dir, 'vis_proof', 'camvid_paper')
    #     os.makedirs(save_path, exist_ok=True)
    #     file_name = f"{name}_iter_{iteration}.png"
    #     plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight', dpi=200)
    #     plt.close()

    # def save_paper_vis(self, img_tensor, target_tensor, masks_tensor, iteration, save_dir, name="sample"):
    #     """
    #     生成 CamVid 论文专用对比图：Input | GT | M1 + Colorbar | M2 + Colorbar | M3 + Colorbar | M4 + Colorbar
    #     """
    #
    #     # 1. 基础图像处理
    #     img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    #     # 反归一化到 0-1
    #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #     H, W = img.shape[:2]
    #
    #     # 2. 生成彩色 Groundtruth
    #     gt_label = target_tensor[0].cpu().numpy()
    #     gt_color = self.decode_segmap(gt_label)
    #
    #     # 3. 掩码插值
    #     masks = F.interpolate(masks_tensor, size=(H, W), mode='bilinear', align_corners=False)[0]
    #     masks = masks.cpu().detach().numpy()  # [4, H, W]
    #
    #     # 4. 创建画布 (1行6列)
    #     # 增加 figsize 宽度以容纳 4 个 colorbar
    #     fig, axs = plt.subplots(1, 6, figsize=(35, 6), facecolor='white')
    #     plt.subplots_adjust(wspace=0.3)  # 适当调大间距，防止 colorbar 和下一张图重叠
    #
    #     # (a) Input Image
    #     axs[0].imshow(img)
    #     axs[0].set_title("Input Image", fontsize=16)
    #     axs[0].axis('off')
    #
    #     # (b) Groundtruth
    #     axs[1].imshow(gt_color)
    #     axs[1].set_title("Groundtruth (GT)", fontsize=16)
    #     axs[1].axis('off')
    #
    #     # (c)-(f) 绘制 4 个 Mask 并添加对齐的 Colorbar
    #     titles = ["Mask 1: Edges", "Mask 2: Textures", "Mask 3: Interiors", "Mask 4: Background"]
    #
    #     for i in range(4):
    #         ax = axs[i + 2]
    #         mask = masks[i]
    #
    #         # 归一化 mask 到 [0, 1] 用于热力图显示
    #         m_min, m_max = mask.min(), mask.max()
    #         if m_max - m_min > 1e-5:
    #             mask_norm = (mask - m_min) / (m_max - m_min)
    #         else:
    #             mask_norm = mask
    #
    #         # 应用 Jet 配色
    #         # 注意：imshow 如果直接传 [H,W] 和 cmap，返回的是可用于 colorbar 的 mappable 对象
    #         ax.imshow(img)  # 先画原图作为背景
    #         im = ax.imshow(mask_norm, cmap='jet', alpha=0.5, vmin=0, vmax=1)  # 叠加半透明热力图
    #         ax.set_title(titles[i], fontsize=16)
    #         ax.axis('off')
    #
    #         # === 核心：添加对齐的状态条 ===
    #         divider = make_axes_locatable(ax)
    #         # 在当前子图右侧开辟一个 5% 宽度的区域放 colorbar，间隔 0.1
    #         cax = divider.append_axes("right", size="5%", pad=0.1)
    #         cbar = plt.colorbar(im, cax=cax)
    #         cbar.ax.tick_params(labelsize=10)  # 调整状态条刻度字体大小
    #
    #     # 5. 存储
    #     save_path = os.path.join(save_dir, 'vis_proof', 'paper')
    #     os.makedirs(save_path, exist_ok=True)
    #     file_name = f"{name}_iter_{iteration}.png"
    #     plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight', dpi=200)
    #     plt.close(fig)

    def save_paper_vis(self, img_tensor, target_tensor, masks_tensor, iteration, save_dir, name="sample"):
        """
        生成论文专用对比图：Input | GT | M1 | M2 | M3 | M4 (带修正的 Colorbar)
        """
        # 1. 基础图像处理
        img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        # 改进的反归一化
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        H, W = img.shape[:2]

        # 2. 生成彩色 Groundtruth
        gt_label = target_tensor[0].cpu().numpy()
        gt_color = self.decode_segmap(gt_label)

        # 3. 掩码插值
        masks = F.interpolate(masks_tensor, size=(H, W), mode='bilinear', align_corners=False)[0]
        masks = masks.cpu().detach().numpy()  # [4, H, W]

        # 4. 创建画布 (1行6列)
        fig, axs = plt.subplots(1, 6, figsize=(38, 6), facecolor='white')
        plt.subplots_adjust(wspace=0.35)  # 留出足够空间给 Colorbar

        # (a) Input Image
        axs[0].imshow(img)
        axs[0].set_title("Input Image", fontsize=18, pad=10)
        axs[0].axis('off')

        # (b) Groundtruth
        axs[1].imshow(gt_color)
        axs[1].set_title("Groundtruth (GT)", fontsize=18, pad=10)
        axs[1].axis('off')

        # 定义标题
        titles = ["Mask 1: Edges", "Mask 2: Textures", "Mask 3: Interiors", "Mask 4: Background"]

        # 设定全局统一映射范围
        norm = Normalize(vmin=0, vmax=1)

        for i in range(4):
            ax = axs[i + 2]
            mask = masks[i]

            # 归一化处理
            m_min, m_max = mask.min(), mask.max()
            if m_max - m_min > 1e-5:
                mask_norm = (mask - m_min) / (m_max - m_min)
            else:
                mask_norm = mask

            # 绘制底层原图
            ax.imshow(img)

            # --- 优化点：提高 Alpha 到 0.6，并确保使用同样的 norm ---
            # 使用 alpha=0.6 让热力图更浓郁，减少背景色的稀释
            im = ax.imshow(mask_norm, cmap='jet', alpha=0.6, norm=norm)
            ax.set_title(titles[i], fontsize=18, pad=10)
            ax.axis('off')

            # === 核心：添加对齐且颜色深沉的 Colorbar ===
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # 明确指定 Colorbar 的 mappable，确保它不受 alpha 影响而变淡
            # 使用独立的 ScalarMappable 确保 Colorbar 永远显示最纯正的 jet 颜色
            mappable = cm.ScalarMappable(norm=norm, cmap='jet')
            cbar = fig.colorbar(mappable, cax=cax)

            cbar.ax.tick_params(labelsize=12)
            # 设置刻度，确保 1.0 处对应最红
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # 5. 存储
        save_path = os.path.join(save_dir, 'vis_proof', 'paper')
        os.makedirs(save_path, exist_ok=True)
        file_name = f"{name}_iter_{iteration}.png"

        # 使用较高的 DPI 确保颜色采样准确
        plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight', dpi=150)

        # 核心：彻底释放资源
        plt.close(fig)
        fig.clf()



    def visualize_paper_masks(self, iteration):
        """ 新增：专门用于生成 CamVid 论文配图的函数 """
        if not self.sentinels:
            return

        # 切换模式
        self.s_model.eval()
        self.criterion_sa.eval()

        with torch.no_grad():
            for s_type, (img, tgt) in self.sentinels.items():
                # 获取教师和学生的输出
                t_out = self.t_model(img)
                s_out = self.s_model(img)

                # 调用 CriterionSA 拿到全部 Masks
                # 注意：确保你的 CriterionSA.forward 在 visualization_mode=True 时返回的是全部 4 个 masks
                masks = self.criterion_sa(
                    feat_S=s_out[-1],
                    feat_T=t_out[-1],
                    feature_transform="STCA_ASCM_M_CKA",
                    activate_s_to_t=True,
                    visualization_mode=True
                )

                # 调用绘图工具
                self.save_paper_vis(
                    img_tensor=img,
                    target_tensor=tgt,
                    masks_tensor=masks,
                    iteration=iteration,
                    save_dir=self.args.log_dir,
                    name=s_type
                )

        # 恢复模式
        self.s_model.train()
        self.criterion_sa.train()



    def adjust_lr(self, base_lr, iter, max_iter, power):
            cur_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            return cur_lr

    # def reduce_tensor(self, tensor):
    #     rt = tensor.clone()
    #     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #     return rt
    #
    # def reduce_mean_tensor(self, tensor):
    #     rt = tensor.clone()
    #     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #     rt /= self.num_gpus
    #     return rt

    def reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        kl_distance = nn.KLDivLoss(reduction='none')
        sm = torch.nn.Softmax(dim=1)
        log_sm = torch.nn.LogSoftmax(dim=1)

        self.s_model.train()

        activate_s_to_t = False  #默认不激活

        # for iteration, (images, targets, _) in enumerate(self.train_loader):
        for iteration, (images, targets, filenames) in enumerate(self.train_loader):
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)

            # # =========================================================
            # # [新增] 逻辑 1: 在第一次迭代时，自动锁定高频和低频样本
            # # =========================================================
            # if iteration <= 50 and self.args.local_rank == 0:
            #     with torch.no_grad():
            #         B = images.shape[0]
            #         variances = images.view(B, -1).var(dim=1)
            #
            #         curr_max_val, curr_max_idx = variances.max(dim=0)
            #         curr_min_val, curr_min_idx = variances.min(dim=0)
            #
            #         # 如果发现了更复杂的图，更新 HighFreq 哨兵
            #         if curr_max_val > self.global_max_var:
            #             self.global_max_var = curr_max_val
            #             self.sentinels['Complex_HighFreq'] = (
            #             images[curr_max_idx:curr_max_idx + 1].clone(), targets[curr_max_idx:curr_max_idx + 1].clone())
            #             logger.info(f"Updated Complex Sentinel (Var: {curr_max_val:.4f})")
            #
            #         # 如果发现了更简单的图（比如纯路面），更新 LowFreq 哨兵
            #         if curr_min_val < self.global_min_var:
            #             self.global_min_var = curr_min_val
            #             self.sentinels['Simple_LowFreq'] = (
            #             images[curr_min_idx:curr_min_idx + 1].clone(), targets[curr_min_idx:curr_min_idx + 1].clone())
            #             logger.info(f"Updated Simple Sentinel (Var: {curr_min_val:.4f})")

            # [修改] 自动抓取逻辑：Simple 限前50次，Complex 限前150次
            # =========================================================
            # 总判断：只要还在前150次内，就需要进这个逻辑块
            if iteration <= 400 and self.args.local_rank == 0:
                with torch.no_grad():
                    # 1. 计算当前Batch的方差
                    B = images.shape[0]
                    variances = images.view(B, -1).var(dim=1)

                    curr_max_val, curr_max_idx = variances.max(dim=0)
                    curr_min_val, curr_min_idx = variances.min(dim=0)

                    # 准备保存工具
                    import torchvision.utils as vutils
                    import os
                    save_root = os.path.join(self.args.log_dir, 'vis_proof', 'Originals')
                    os.makedirs(save_root, exist_ok=True)

                    # --- A. 寻找更复杂的样本 (Complex) ---
                    # 条件：一直找，直到第  次迭代结束
                    if iteration <= 400:
                        if curr_max_val > self.global_max_var:
                            self.global_max_var = curr_max_val

                            self.sentinels['Complex_HighFreq'] = (
                                images[curr_max_idx:curr_max_idx + 1].clone(),
                                targets[curr_max_idx:curr_max_idx + 1].clone()
                            )

                            file_complex = filenames[curr_max_idx]
                            logger.info(
                                f"[Iter {iteration}] Found New Max Variance: {curr_max_val:.4f} -> {file_complex}")

                            # 保存 Complex 图片
                            img_save = images[curr_max_idx].clone().detach()
                            img_save = (img_save - img_save.min()) / (img_save.max() - img_save.min())
                            vutils.save_image(img_save, os.path.join(save_root, f"Best_Complex_HighFreq.png"))

                    # --- B. 寻找更简单的样本 (Simple) ---
                    # 条件：只在前 50 次迭代内找，50次后不再更新 Simple
                    if iteration <= 50:
                        if curr_min_val < self.global_min_var:
                            self.global_min_var = curr_min_val

                            self.sentinels['Simple_LowFreq'] = (
                                images[curr_min_idx:curr_min_idx + 1].clone(),
                                targets[curr_min_idx:curr_min_idx + 1].clone()
                            )

                            file_simple = filenames[curr_min_idx]
                            logger.info(
                                f"[Iter {iteration}] Found New Min Variance: {curr_min_val:.4f} -> {file_simple}")

                            # 保存 Simple 图片
                            img_save = images[curr_min_idx].clone().detach()
                            img_save = (img_save - img_save.min()) / (img_save.max() - img_save.min())
                            vutils.save_image(img_save, os.path.join(save_root, f"Best_Simple_LowFreq.png"))


            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            warmup_iter = args.warmup_iter

            # VGD Stage
            if iteration < warmup_iter:
                activate_s_to_t = False  # 在VGD阶段，强制不激活S->T
                if self.args.aux:
                    task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1],
                                                                                             targets)
                else:
                    task_loss = self.criterion(s_outputs[0], targets)

                # borrow from https://github.com/layumi/Seg-Uncertainty/blob/master/trainer_ms_variance.py#L166
                # training only easy samples
                variance = torch.sum(kl_distance(log_sm(t_outputs[0]), sm(t_outputs[1])), dim=1)
                exp_variance = torch.exp(-variance)
                exp_variance_scale = torch.unsqueeze(exp_variance, 1)

                # exp_variance_scale = F.interpolate(exp_variance_scale, (512, 1024), mode='bilinear',
                #                                    align_corners=True)
                # 将 variance map 上采样到当前 batch 的目标尺寸，而不是写死为 cityscapes 的 512x1024
                exp_variance_scale = F.interpolate(exp_variance_scale, (targets.size(1), targets.size(2)),
                                                   mode='bilinear',
                                                   align_corners=True)

                exp_variance_scale = exp_variance_scale.squeeze()

                task_loss = torch.mean(task_loss * exp_variance_scale)

            # EDD Stage
            else:

                if self.args.aux:
                    task_loss_main, _, s_to_t_active_main = self.OHEMcriterion.forward_with_teacher_student(
                        s_outputs[0],
                        t_outputs[0],
                        targets)

                    task_loss_aux, _, _ = self.OHEMcriterion.forward_with_teacher_student(s_outputs[1],
                                                                                          t_outputs[1],
                                                                                          targets)
                    task_loss = task_loss_main + 0.4 * task_loss_aux

                    # 更新当前迭代的激活状态
                    activate_s_to_t = s_to_t_active_main

                else:
                    task_loss, _, s_to_t_active = self.OHEMcriterion.forward_with_teacher_student(s_outputs[0],
                                                                                                  t_outputs[0],
                                                                                                  targets)

                    activate_s_to_t = s_to_t_active

            cwd_logit_loss = torch.tensor(0.).cuda()
            sa_loss = torch.tensor(0.).cuda()

            # s_outputs[-1] : features
            # s_outputs[0] : logits
            if self.args.lambda_kd != 0.:
                kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], t_outputs[0])
            if self.args.lambda_cwd_logit != 0:
                cwd_logit_loss = self.args.lambda_cwd_logit * self.criterion_cwd(s_outputs[0], t_outputs[0])

            # if getattr(self.args, 'lambda_stca', 0.) != 0.:
            #     # CriterionSA 在 STCA_ASCM_M_CKA 模式下返回单个标量 loss
            #     loss_stca, freq_loss = self.criterion_sa(s_outputs[-1], t_outputs[-1], feature_transform="STCA_ASCM_M_CKA",
            #                                   activate_s_to_t=activate_s_to_t)
            #     sa_loss = self.args.lambda_stca * loss_stca
            # if getattr(self.args, 'lambda_stca', 0.) != 0.:
            #     # [关键修改] 调用 forward 获取 Loss 和 Stats
            #     loss_stca, (sigma_mean, gamma_mean) = self.criterion_sa(
            #         s_outputs[-1], t_outputs[-1],
            #         feature_transform="STCA_ASCM_M_CKA",
            #         activate_s_to_t=activate_s_to_t
            #     )
            #     sa_loss = self.args.lambda_stca * loss_stca
            #     avg_sigma = sigma_mean
            #     avg_gamma = gamma_mean
            if getattr(self.args, 'lambda_stca', 0.) != 0.:

                loss_stca,  mask_details= self.criterion_sa(
                    s_outputs[-1], t_outputs[-1],
                    feature_transform="STCA_ASCM_M_CKA",
                    activate_s_to_t=activate_s_to_t
                )
                sa_loss = self.args.lambda_stca * loss_stca
                avg_sigma = torch.tensor([m['mask_0_sigma'] for m in mask_details if 'mask_0_sigma' in m]).mean()
                avg_gamma = torch.tensor([m['mask_0_gamma'] for m in mask_details if 'mask_0_gamma' in m]).mean()
            else:
                if self.args.lambda_cam != 0 and self.args.lambda_pam == 0:
                    loss_CAM = self.criterion_sa(s_outputs[-1], t_outputs[-1], feature_transform="CAM_CKA")
                    sa_loss = self.args.lambda_cam * loss_CAM
                if self.args.lambda_cam == 0 and self.args.lambda_pam != 0:
                    loss_PAM = self.criterion_sa(s_outputs[-1], t_outputs[-1], feature_transform="gridPAM_CKA")
                    sa_loss = self.args.lambda_pam * loss_PAM
                if self.args.lambda_cam != 0 and self.args.lambda_pam != 0:
                    loss_CAM, loss_PAM = self.criterion_sa(s_outputs[-1], t_outputs[-1],
                                                           feature_transform="separately_CAMgridPAM_CKA")
                    sa_loss = self.args.lambda_cam * loss_CAM + self.args.lambda_pam * loss_PAM


            losses = sa_loss + cwd_logit_loss + task_loss

            lr = self.adjust_lr(base_lr=args.lr, iter=iteration - 1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            cwd_logit_loss_reduced = self.reduce_mean_tensor(cwd_logit_loss)
            sa_loss_reduced = self.reduce_mean_tensor(sa_loss)

            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                s_to_t_status = "ON" if activate_s_to_t else "OFF"
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f}" \
                    "|| cwd_logit_loss: {:.4f} || sa_loss: {:.4f} || |Sig: {:.2f} Gam: {:.2f} || S->T: {} " \
                    "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        cwd_logit_loss_reduced.item(),
                        sa_loss_reduced.item(),
                        avg_sigma.item(),  # 打印 Sigma
                        avg_gamma.item(),  # 打印 Gamma
                        s_to_t_status,  # 在日志中打印S->T状态
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            # =========================================================
            # [新增] 逻辑 2: 定期 (每100iter) 调用探测函数生成可视化图
            # =========================================================
            # if self.args.local_rank == 0 and iteration % 200 == 0:
            #     self.probe_training_sentinels(iteration)

            if self.args.local_rank == 0 and iteration % 200 == 0:
                # [新增调用] 专门生成用于论文的彩色对比图
                self.visualize_paper_masks(iteration)


            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)

            # pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, mDice = self.metric.get()  # <--- 修改这里

            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
            logger.info(
                "Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}, mDice: {:.3f}".format(i + 1, pixAcc, mIoU,
                                                                                              mDice))  # <--- 修改这里

        if self.num_gpus > 1:
            # ... (分布式计算部分保持不变)
            # 注意：分布式计算部分目前只计算了 pixAcc 和 mIoU，如果需要精确的分布式 mDice，
            # 也需要像 total_inter 和 total_union 一样同步 total_pred_area 和 total_lab_area。
            # 为简单起见，这里我们只在主进程中报告最终的聚合结果。
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            # 新增
            sum_total_pred_area = torch.tensor(self.metric.total_pred_area).cuda().to(args.local_rank)
            sum_total_lab_area = torch.tensor(self.metric.total_lab_area).cuda().to(args.local_rank)

            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            # 新增
            sum_total_pred_area = self.reduce_tensor(sum_total_pred_area)
            sum_total_lab_area = self.reduce_tensor(sum_total_lab_area)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            # 新增
            Dice = (2.0 * sum_total_inter) / (2.220446049250313e-16 + sum_total_pred_area + sum_total_lab_area)
            mDice = Dice.mean().item()

            # logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #     pixAcc.item() * 100, mIoU * 100))

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}, mDice: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100, mDice * 100))  # <--- 修改这里

        else:
            # pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, mDice = self.metric.get()  # <--- 修改这里

        # logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
        #     pixAcc.item() * 100 if isinstance(pixAcc, torch.Tensor) else pixAcc * 100,
        #     mIoU * 100))

        logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}, mDice: {:.3f}".format(
            pixAcc.item() * 100 if isinstance(pixAcc, torch.Tensor) else pixAcc * 100,
            mIoU * 100,
            mDice * 100))  # <--- 修改这里

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        new_pixAcc = pixAcc.item() if isinstance(pixAcc, torch.Tensor) else pixAcc
        if new_pixAcc > self.best_pixAcc:
            self.best_pixAcc = new_pixAcc

        # 新增：更新并记录最佳 mDice
        if mDice > self.best_mDice:
            self.best_mDice = mDice

        # logger.info("Best pixAcc: {:.3f}, Best mIoU: {:.3f}".format(
        #     self.best_pixAcc * 100, self.best_pred * 100))

        # 修改：在日志中打印最佳 mDice
        logger.info("Best pixAcc: {:.3f}, Best mIoU: {:.3f}, Best mDice: {:.3f}".format(
            self.best_pixAcc * 100, self.best_pred * 100, self.best_mDice * 100))

        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""
    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'CSKD_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'CSKD_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        print("if args.seed is not None:")
        set_seed(args.seed)

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
