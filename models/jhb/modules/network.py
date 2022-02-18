# Total3D: model loader
# author: ynie
# date: Feb, 2020

from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from torch import nn
import numpy as np


@METHODS.register_module
class JHB(BaseNetwork):
    '''
    对应我们的JHB.yaml中 method: JHB
    '''
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        #all we have is the only phase which is jhb_classify
        if cfg.config[cfg.config['mode']]['phase'] in ['jhb_classify']:# jfr 去掉joint的layout estimation
            phase_names += ['jhb_classify']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name in phase_names:
            if phase_name not in cfg.config['model'].keys():
                continue
            net_spec = cfg.config['model'][phase_name]# is still a dictionary
            method_name = net_spec['method']# to get the name of the network
            '''
            model:
                jhb_classify:
                    method: HJHBNet
                    loss: HJHBLoss
            '''
            # load specific optimizer parameters
            print('method_name: ',method_name)
            if MODULES is None:
                print('MODULES is None')
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            bfsubnet=MODULES.get(method_name)
            if bfsubnet is None:
                print("bfsubnet is None")
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)
            #出现了self.jhb_classify

            '''load corresponding loss functions'''
            # jhb_classify_loss
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1), cfg.config))

        '''freeze submodules or not'''
        #self.freeze_modules(cfg)

    def forward(self, data):
        all_output = {}

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['jhb_classify']:
            cls_result = self.jhb_classify(data['image'])
            cls_output = {'cls_result':cls_result}
            all_output.update(cls_output)#here we can stop the network pass and the update of the output and it is alright

        if all_output:
            return all_output
        else:
            raise NotImplementedError

    def loss(self, est_data, gt_data, mode='train'):
        '''
        calculate loss of est_out given gt_out.
        注意输出形式（很可能是三个置信度，和输入形式之间区别，因此最好把输入改成one hot形式，随便在哪里都可以）
        '''
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['jhb_classify']:
            cls_loss=self.jhb_classify_loss(est_data,gt_data)
            total_loss = sum(cls_loss.values())
            '''
            return {'cls_loss': cls_loss}
            cls_loss={'cls_loss': cls_loss}
            之前下面返回 'total': cls_loss
            是个字典
            要改
            '''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'jhb_classify':
            return {'total':total_loss, **cls_loss}
        else:
            raise NotImplementedError
