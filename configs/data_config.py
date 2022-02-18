"""
Created on May, 2019

@author: Yinyu Nie

Data configurations.

"""


class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

num_samples_on_each_model = 5000
n_object_per_image_in_training = 8

import os
import numpy as np
import pickle

FUTURE_TO_NYU37_CLS_MAPPING={1:"cabinet", 2:"night_stand", 3:"bookshelf", 4:"shelves", 5: "table",
                6:"table", 7:"dresser", 8: "cabinet", 9: "cabinet", 10: "cabinet", 11: "shelves",
                             12:"table", 13:"bed", 14:"bed" , 15: "bed" ,16: "bed" , 17: "bed" , 18:"chair",
                             19: "chair", 20: "chair", 21: "chair", 22: "chair", 23:"table", 24:"table",
                             25: "table", 26: "sofa", 27: "sofa", 28: "sofa", 29: "sofa", 30: "sofa", 31: "sofa",
                             32: "chair", 33: "lamp", 34:"lamp", 0:"void"}#bug 1
#jfr

NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

NYU37_TO_PIX3D_CLS_MAPPING = {0:0, 1:0, 2:0, 3:8, 4:1, 5:3, 6:5, 7:6, 8:8, 9:2, 10:2, 11:0, 12:0, 13:2, 14:4,
                              15:2, 16:2, 17:8, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:8, 25:8, 26:0, 27:0, 28:0,
                              29:8, 30:8, 31:0, 32:8, 33:0, 34:0, 35:0, 36:0, 37:8}

RECON_3D_CLS = [3,4,5,6,7,8,10,14,15,17,24,25,29,30,32]

number_pnts_on_template = 2562

pix3d_n_classes = 9

cls_reg_ratio = 10
obj_cam_ratio = 1

class Config(object):
    def __init__(self, dataset):
        """
        Configuration of data paths.
        """
        self.dataset = dataset

        if self.dataset == '3d-future':
            self.metadata_path = './data/sunrgbd'
            self.train_test_data_path = os.path.join(self.metadata_path, '3d-future_train_test_data')
            #self.__size_avg_path = './data/3d-future/test1/size_avg_category_future.pkl'
            self.__size_avg_path = os.path.join(self.metadata_path, 'preprocessed/size_avg_category_future.pkl')
            self.__layout_avg_file = os.path.join(self.metadata_path, 'preprocessed/layout_avg_file.pkl')
            self.bins = self.__initiate_bins()
            self.evaluation_path = './evaluation/sunrgbd'
            if not os.path.exists(self.train_test_data_path):
                os.mkdir(self.train_test_data_path)

    def __initiate_bins(self):
        bin = {}

        if self.dataset == '3d-future':
            # there are faithful priors for layout locations, we can use it for regression.
            '''
            一些关于布局的先验知识，可能就是在于应该分成几份，每份多大？
            '''
            if os.path.exists(self.__layout_avg_file):#有layout_avg_file
                with open(self.__layout_avg_file, 'rb') as file:
                    layout_avg_dict = pickle.load(file)#用这个方式去转换为字典格式，具体可以在这里输出看一下情况
            '''layout orientation bin'''
            NUM_LAYOUT_ORI_BIN = 2
            ORI_LAYOUT_BIN_WIDTH = np.pi / 4
            bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in range(NUM_LAYOUT_ORI_BIN)]

            '''camera bin'''
            PITCH_NUMBER_BINS = 2
            PITCH_WIDTH = 40 * np.pi / 180
            ROLL_NUMBER_BINS = 2
            ROLL_WIDTH = 20 * np.pi / 180

            # pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
            bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                                i in range(PITCH_NUMBER_BINS)]
            # roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
            bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                               range(ROLL_NUMBER_BINS)]

            '''bbox orin, size and centroid bin'''
            # orientation bin
            NUM_ORI_BIN = 6#一共分成6个象限（区域，bin这个词在统计学上有含义，可以词典搜索看到就是）
            '''
            ORI_BIN_WIDTH = float(360 / NUM_ORI_BIN)
            '''
            ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN) # 60 degrees width for each bin. 一共象限60度
            # orientation bin ranges from -np.pi to np.pi.
            bin['ori_bin'] = [[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                              in range(NUM_ORI_BIN)]
            '''
            i=0: [(0-3)*60度, (0-3+1)*60度]=[-180, -120]
            i=1: [(1-3)*60度, (1-3+1)*60度]=[-120, -60]
            .
            .
            .
            i=5: [(5-3)*60度, (5-3+1)*60度]=[120, 180]
            综上所述:
            bin['ori_bin'].shape=(6,2)
            bin['ori_bin']=[[-180,-120],[-120,-60],...,[120,180]]
            '''

            if os.path.exists(self.__size_avg_path):
                with open(self.__size_avg_path, 'rb') as file:
                    avg_size = pickle.load(file)
                    #print("avg_size: ",avg_size)
            else:
                raise IOError('No object average size file in %s. Please check.' % (self.__size_avg_path))

            bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])
            #print("bin['avg_size']: ",bin['avg_size'])#jfr
            # for each object bbox, the distance between camera and object centroid will be estimated.
            #added by jfr
            NUM_DEPTH_BIN = 6#深度（距离）分成6组，每组的变化限度是6
            DEPTH_WIDTH = 1.0
            #DEPTH_WIDTH = 1.0
            # centroid_bin = [0, 6]
            bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in range(2,8)]
        else:
            raise NameError('Please specify a correct dataset name.')

        return bin
