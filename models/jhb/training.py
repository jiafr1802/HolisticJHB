# Trainer for Total3D.
# author: ynie
# date: Feb, 2020
import os
from models.training import BaseTrainer
import torch
from net_utils.libs import get_rotation_matrix_gt, get_mask_status
from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING, FUTURE_TO_NYU37_CLS_MAPPING, NYU40CLASSES#jfr

class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''
    def eval_step(self, data):# 这个只是在test的时候才会被调用对吗
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        with torch.no_grad():
            '''
            https://blog.csdn.net/gdengden/article/details/107778444?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EHighlightScore-1.queryctrv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EHighlightScore-1.queryctrv2&utm_relevant_index=2
            除此之外，注意pytorch在test时，一定要加上
            '''
            # test process
            loss = self.compute_loss(data,mode='test')#jfr
        loss['total'] = loss['total'].item()
        '''
        呼应network.py中 def loss中的：
        return {'total':cls_loss}
        '''
        return loss

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        pass

    def to_device(self, data):
        device = self.device

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['jhb_classify']:
            '''calculate loss from camera and layout estimation'''
            image = data['image'].to(device)
            image = data['image'].to(device)
            label = data['label'].to(device)
            cls_input = {'image':image,'label': label}

        '''output data'''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'jhb_classify':
            return cls_input
        else:
            raise NotImplementedError

    def compute_loss(self, data, mode='train'):#这样真正train的地方调用compute loss就不需要改了，全方位调查
        #jfr
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''computer losses'''
        loss = self.net.loss(est_data, data, mode)#jfr => network.py中jhb类中的一个方法：loss
        '''
        what if 
        '''
        return loss
