# Tester for Total3D
# author: ynie
# date: April, 2020
import os
from models.testing import BaseTester
from .training import Trainer
import torch
from scipy.io import savemat
import numpy as np
from ..loss import HJHBLoss


class Tester(BaseTester, Trainer):
    '''
    ??? 是要继承两个类吗？？？
    同时继承BaseTester和Trainer
    '''
    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)
        self.jhb_classify_loss = HJHBLoss()

    def to_device(self, data):
        #data_output = super(Tester, self).to_device(data)
        '''
        所以上面调用的to_device其实是Trainer里面的to_device
        我们的根本问题还是要回归到python如何继承两个类
        '''
        #return {**data_output}
        #以下 jfr
        device = self.device

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['jhb_classify']:
            '''calculate loss from camera and layout estimation'''
            '''
            忘记录入Label
            '''
            image = data['image'].to(device)
            label = data['label'].to(device)
            cls_input = {'image':image,'label': label,'file_path': data['file_path']}#jfr

        '''output data'''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'jhb_classify':
            return cls_input
        else:
            raise NotImplementedError

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        metrics = {}
        # welcome
        '''
        我觉得此处不同于训练部分的损失函数的计算，原因在于此处我们要知道的是分类准确率
        因此要将我们预测的类的结果和我们的gt类的结果同时输出，方便后面计算准确率
        '''
        '''
        先清楚est_data的格式
            est_data => cls_output = {'cls_result':cls_result}
                cls_result:  tensor([[-0.0974, -0.0087,  0.0796],
                            [-0.1888, -0.0710,  0.1040]], device='cuda:0', grad_fn=<AddmmBackward>)
                batch_size x 3 
        再利用est_data的格式转换成预测类别
        pre_cls=torch.argmax(est_data['cls_result'], 1)
        可以输出看一下
        返回pre_cls和gt_cls
        总之，自己设定格式，自己搞清楚就OK  
        '''
        loss = self.net.loss(est_data, gt_data)#jfr => network.py中jhb类中的一个方法：loss
        metrics=loss
        est_data=torch.argmax(est_data['cls_result'], 1)
        est_data=est_data.cpu().numpy().tolist()
        file_path=gt_data['file_path']#jfr
        gt_data= gt_data['label'].cpu().numpy().tolist()
        for i in range(len(est_data)):
            self.cfg.log_string("cls: gt is %s, est is %s, file path is %s" % (gt_data[i], est_data[i], file_path[i]))

        '''
        这样可能pre和gt都是
        '''
        '''
        no! 注意 metrics必须是可以求平均的东西
        因为后面出去有对loss avg
        所以metrics应该返回的还是entropy_loss类似的东西
        当然要和分类问题契合
        这个也是为了和train loss对比吧（去看train部分）
        那么pre和gt就在这里输出
        '''
        return metrics#此处输出也就是test_step的输出

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        loss = self.get_metric_values(est_data, data)
        '''
        loss记录真实和预测类别
        loss['pre']
        loss['gt'] 
        '''
        return loss

