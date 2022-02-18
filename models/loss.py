# loss function library.
# author: ynie
# date: Feb, 2020
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.data_config import cls_reg_ratio
from models.registers import LOSSES
import numpy as np

cls_criterion = nn.CrossEntropyLoss(reduction='mean')

def get_cls_loss(cls_result, cls_gt):
    print("cls_result: ",cls_result)
    print("cls_gt: ",cls_gt)
    cls_loss = cls_criterion(cls_result, cls_gt.long())#jfr
    '''
    因为CrossEntropyLoss的源码里写的dtype就是long
    '''
    return cls_loss

class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1, config=None):
        '''initialize loss module'''
        self.weight = weight
        self.config = config

@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''
    def __call__(self, loss):
        return self.weight * torch.mean(loss)

@LOSSES.register_module
class HJHBLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        '''
        :param est_data: 三个类的置信度？
        :param gt_data: one-hot ？
        :return: 为什么是字典形式我现在也不知道，之前是因为有多项多种损失，现在为了方便吧
        '''
        print('est_data: ',est_data)
        print('gt_data: ',gt_data)
        cls_loss=get_cls_loss(est_data['cls_result'],gt_data['label'])#jfr
        return {'cls_loss': cls_loss}
