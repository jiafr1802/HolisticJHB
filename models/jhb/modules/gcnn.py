# SGCN
# author: chengzhang

import torch
import torch.nn as nn
from models.registers import MODULES
from configs.data_config import NYU40CLASSES, pix3d_n_classes
import torch.nn.functional as F
from net_utils.libs import get_bdb_form_from_corners, recover_points_to_world_sys, \
    get_rotation_matix_result, get_bdb_3d_result, recover_points_to_obj_sys


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # Nobj x Nrel Nrel x dim
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg


class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
    def forward(self, target, source):
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (subject) from rel
        '''
        where is the di_rel come from 
        '''
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (object)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj)) # obj from obj

    def forward(self, target, source, attention, unit_id):
        '''
        :param target:
        :param source:
        :param attention:
        :param unit_id: 这个可以告诉我们使用上面我们搞到的5个[0-4]之中的哪一个核子！
        :return:
        '''
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj)) # obj from others
        self.update_units.append(_Update_Unit(dim_rel)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


@MODULES.register_module
class GCNN(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(GCNN, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''configs and params'''
        self.cfg = cfg
        self.lo_features = cfg.config['model']['output_adjust']['lo_features']# they are just dict !!!
        self.obj_features = cfg.config['model']['output_adjust']['obj_features']
        '''
        obj_features: ['size_cls', 'cls_codes', 'size_reg_result', 'ori_reg_result', 'ori_cls_result',
                   'centroid_reg_result', 'centroid_cls_result', 'offset_2D_result',
                   'analytic_code', 'odn_arfeatures', 'blob_center']
        目前来说只有 analytic_code和 obn_arfeatures, blob_center是陌生的，但是很快
        大哥 analytic_code v = data['structured_implicit'].analytic_code
        in mesh_reconstruction.py -> forward: 
        structured_implicit = StructuredImplicit(config=self.cfg.config, **structured_implicit, net=self)
        and the structured_implicit is returned as:
        return_dict['structured_implicit'] = structured_implicit.dict()# also a dict
        the dict is:
            def dict(self):
        return {'constant': self.constants, 'radius': self.radii, 'center': self.centers, 'iparam': self.iparams}
        
        and StructuredImplicit has the [property] analytic_code
        it is spetial means that we can get it anytime as a property 
        so it is very clear 
        二哥 odn_arfeatures is the output of the attention module so it is alright 
            
        三哥 blob_center is the center points of the mesh's points 
        '''
        self.rel_features = cfg.config['model']['output_adjust']['rel_features']
        '''
        rel_features: ['g_features', 'bdb2D_pos'] all we alright had yeah 
        '''
        feature_dim = cfg.config['model']['output_adjust']['feature_dim']
        self.feat_update_step = cfg.config['model']['output_adjust']['feat_update_step']# step = 4 I guess
        self.res_output = cfg.config['model']['output_adjust'].get('res_output', False)
        self.feat_update_group = cfg.config['model']['output_adjust'].get('feat_update_group', 1 )# group operation?
        self.res_group = cfg.config['model']['output_adjust'].get('res_group', False)

        self.feature_length = {
            'size_cls': len(NYU40CLASSES), 'cls_codes': pix3d_n_classes,
            'bdb2D_pos': 4, 'g_features': 32, 'mgn_afeature': 1024, 'K': 3,
            'pitch_reg_result': 2, 'roll_reg_result': 2, 'pitch_cls_result': 2,
            'roll_cls_result': 2, 'lo_ori_reg_result': 2, 'lo_ori_cls_result': 2,
            'lo_centroid_result': 3, 'lo_coeffs_result': 3, 'lo_afeatures': 2048,
            'size_reg_result': 3, 'ori_reg_result': 6, 'ori_cls_result': 6,
            'centroid_reg_result': 6, 'centroid_cls_result': 6, 'offset_2D_result': 2,
            'odn_afeature': 2048, 'odn_rfeatures': 2048, 'odn_arfeatures': 2048,
            'ldif_afeature': cfg.config['model']['mesh_reconstruction'].get('bottleneck_size', None),
            'analytic_code': cfg.config['model']['mesh_reconstruction'].get('analytic_code_len', None),
            'blob_center': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
                           + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) * 3,
            'ldif_phy': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
                           + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) // 2,
            'structured_implicit_vector': cfg.config['model']['mesh_reconstruction'].get('structured_implicit_vector_len', None)
        }
        obj_features_len = sum([self.feature_length[k] for k in self.obj_features])
        rel_features_len = sum([self.feature_length[k] for k in self.rel_features]) * 2
        lo_features_len = sum([self.feature_length[k] for k in self.lo_features])

        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])

        '''modules'''
        # feature embedding (from graph-rcnn)
        self.obj_embedding = nn.Sequential(
            nn.Linear(obj_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(rel_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.lo_embedding = nn.Sequential(
            nn.Linear(lo_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )

        # graph message passing (from graph-rcnn)
        if self.feat_update_step > 0:
            self.gcn_collect_feat = nn.ModuleList([
                _GraphConvolutionLayer_Collect(feature_dim, feature_dim) for i in range(self.feat_update_group)])
            self.gcn_update_feat = nn.ModuleList([
                _GraphConvolutionLayer_Update(feature_dim, feature_dim) for i in range(self.feat_update_group)])

        # feature to output (from Total3D object_detection)
        # branch to predict the size
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc4 = nn.Linear(feature_dim // 2, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_centroid = nn.Linear(feature_dim // 2, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_off_2 = nn.Linear(feature_dim // 2, 2)

        # feature to output (from Total3D layout_estimation)
        # we don't need that
        self.fc_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_2 = nn.Linear(feature_dim // 2, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        # fc for layout
        #no need
        self.fc_layout = nn.Linear(feature_dim, feature_dim)
        # for layout orientation
        #no need
        self.fc_3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_4 = nn.Linear(feature_dim // 2, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        #no need
        self.fc_5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_6 = nn.Linear(feature_dim // 2, 6)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _K2feature(self, K):
        camKs = K.reshape(K.shape[0], -1)
        camKs = camKs.index_select(1, torch.tensor([0, 2, 4, 5], device=camKs.device))
        camKs = camKs[:, :3] / camKs[:, 3:]
        return camKs

    def _get_bdb3D_form(self, data):
        # camera orientation for evaluation
        cam_R_out = get_rotation_matix_result(self.cfg.bins_tensor,
                                              torch.argmax(data['pitch_cls_result'], 1),
                                              data['pitch_reg_result'],
                                              torch.argmax(data['roll_cls_result'], 1),
                                              data['roll_reg_result'])

        # projected center
        P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 - (
                data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * data['offset_2D_result'][:, 0],
                                (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 - (
                                        data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result, _ = get_bdb_3d_result(self.cfg.bins_tensor,
                                            torch.argmax(data['ori_cls_result'], 1),
                                            data['ori_reg_result'],
                                            torch.argmax(data['centroid_cls_result'], 1),
                                            data['centroid_reg_result'],
                                            data['size_cls'],
                                            data['size_reg_result'],
                                            P_result,
                                            data['K'],
                                            cam_R_out,
                                            data['split'])
        bdb3D_form = get_bdb_form_from_corners(bdb3D_result)
        '''
        oh yeah I know it is just the parameters of the bdb3d 
            return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}
            is the return form and actually I omit it in my project 
        '''
        return bdb3D_form

    def _get_object_features(self, data, type):
        features = []
        keys = self.obj_features if type == 'obj' else self.rel_features
        '''
        rel_features:
        ['g_features', 'bdb2D_pos']
        yes this two one, we have had both the a&b and a&a type 
        '''
        for k in keys:
            if k in ['size_cls', 'cls_codes', 'size_reg_result', 'ori_reg_result', 'ori_cls_result',
                     'centroid_reg_result', 'centroid_cls_result', 'offset_2D_result',
                     'ldif_afeature', 'mgn_afeature', 'odn_afeature', 'odn_rfeatures', 'odn_arfeatures']:
                v = data[k]
            elif k == 'g_features':
                assert type == 'rel'
                v = data[k]
            elif k == 'bdb2D_pos':
                v = data[k].clone()
                center_inds = data['K'][:, :2, 2]# next is like a normalization yeahhhhhh
                '''
                [:2] means the first two so is the cx and cy yeah 
                '''
                for center_ind, (start, end) in zip(center_inds, data['split']):# both a&b and a&a yes
                    for i in range(start.item(), end.item()):
                        v[i][0] = (v[i][0] - center_ind[0]) / center_ind[0] # like a normalization
                        v[i][2] = (v[i][2] - center_ind[0]) / center_ind[0]
                        v[i][1] = (v[i][1] - center_ind[1]) / center_ind[1]
                        v[i][3] = (v[i][3] - center_ind[1]) / center_ind[1]
            elif k == 'K':
                camKs = self._K2feature(data[k])# camera feature alright
                v = []
                for i, (start, end) in enumerate(data['split']):
                    v.append(camKs[i:i+1, :].expand(end-start, -1))
                v = torch.cat(v, 0)
            elif k in ['analytic_code', 'structured_implicit_vector', 'blob_center']:
                if k == 'analytic_code':#also the output from mesh reconstruction
                    v = data['structured_implicit'].analytic_code
                    '''
                    return_dict['structured_implicit'] = structured_implicit.dict()
                    '''
                elif k == 'structured_implicit_vector':
                    v = data['structured_implicit'].vector
                elif k == 'blob_center':
                    '''
                    this is odd since I don't trust the transfer of coordinate system of them 
                    '''
                    # get world_sys points from blob centers
                    bdb3D_form = self._get_bdb3D_form(data)# the basis and such like that
                    #we can omit the bdb3D_form to get the center (we can use the angle and rotation and like that
                    centers = data['structured_implicit'].all_centers.clone()
                    centers[:, :, 2] *= -1
                    v = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                    '''
                    this is a disaster if we abuse it but i won't 
                    we can figure out what is exactly this part want and to feed it 
                    
                    recover_points_to_world_sys: 
                     Get 3D point cloud from mesh with estimated position and orientation.
                    :param bdb3D: 3D object bounding boxes with keys ['coeffs', 'basis', 'centroid'].
                    :param mesh_coordinates: Number_of_objects x Number_of_points x 3.
                    :return: points on world system
                    why we need the point clouds, alright 
                    I can finish the transformation and it is easy for me to change from object system to the world system 
                    '''
                v = v.reshape([v.shape[0], -1])
            elif k == 'ldif_phy':
                assert type == 'rel'
                bdb3D_form = self._get_bdb3D_form(data)
                structured_implicit = data['structured_implicit']
                ldif_center, ldif_coef = data['obj_center'], data['obj_coef']

                # get world_sys points from blob centers
                centers = data['structured_implicit'].all_centers.clone()
                centers[:, :, 2] *= -1
                obj_samples = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                element_count = centers.shape[1]

                # put points to other objects' coor
                max_sample_points = (data['split'][:, 1] - data['split'][:, 0]).max() * element_count
                other_obj_samples = torch.zeros([len(obj_samples), max_sample_points, 3], device=centers.device)
                for start, end in data['split']:
                    other_obj_sample = obj_samples[start:end].reshape(1, -1, 3).expand(end - start, -1, -1)
                    other_obj_samples[start:end, :other_obj_sample.shape[1]] = other_obj_sample
                other_obj_samples = recover_points_to_obj_sys(bdb3D_form, other_obj_samples, ldif_center, ldif_coef)
                '''
                change it !!!!!! 
                this need to be done 
                recover_points_to_obj_sys
                put points to other objects' coor
                not too hard 
                
                '''
                other_obj_samples[:, :, 2] *= -1

                # get sdf of blob centers from other objects
                est_sdf = data['mgn'](
                    samples=other_obj_samples,
                    structured_implicit=structured_implicit.dict(),
                    apply_class_transfer=False,
                )['global_decisions'] + 0.07

                # reshape into relation features
                v = [est_sdf[start:end, :(end - start) * element_count].reshape(-1, element_count)
                     for start, end in data['split']]
                v = torch.cat(v)

            else:
                raise NotImplementedError

            if type == 'obj' or k in ('g_features', 'ldif_phy'):
                features.append(v)# it is the core
            else:# type='rel' and k= bdb2D_pos
                features_rel = []
                for start, end in data['split']:
                    features_rel.append(torch.stack(
                        [torch.cat([loc1, loc2], -1)
                         for loc1 in v[start:end]
                         for loc2 in v[start:end]]))
                features.append(torch.cat(features_rel, 0))
        return torch.cat(features, -1)

    def _get_layout_features(self, data):
        features = []
        keys = self.lo_features
        for k in keys:
            if k in ['pitch_reg_result', 'roll_reg_result', 'pitch_cls_result',
                     'roll_cls_result', 'lo_ori_reg_result', 'lo_ori_cls_result',
                     'lo_centroid_result', 'lo_coeffs_result', 'lo_afeatures']:
                v = data[k]
            elif k == 'K':
                v = self._K2feature(data[k])
            else:
                raise NotImplementedError

            features.append(v)
        return torch.cat(features, -1)

    def _get_map(self, data):# need to be change because of the lack of layout
        device = data['g_features'].device
        split = data['split']
        obj_num = split[-1][-1] + split.shape[0]  # number of objects and layouts
        # ！！！ omit the layout and it is easy
        #obj_num = split[-1][-1] we can print it out
        obj_obj_map = torch.zeros([obj_num, obj_num])  # mapping of obj/lo vertices with connections

        rel_inds = []  # indexes of vertices connected by relation nodes -> what  注意是indexes！
        rel_masks = []  # mask of relation features for obj/lo vertices connected by relation nodes -> what and ok
        obj_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of object vertices ok
        lo_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of layout vertices ok
        for lo_index, (start, end) in enumerate(split):#lo_index is the index of the room
            start = start + lo_index  # each subgraph has Ni object vertices and 1 layout vertex
            end = end + lo_index + 1  # consider layout vertex, Ni + 1 vertices in total
            #start=start, end = end
            '''
            因为layout的加入，我们的一个大图（多个image构成），首先是几个小图，一个图是一个image
            然后一个图中的节点数是 num_obj+1 
            '''
            obj_obj_map[start:end, start:end] = 1  # each subgraph is a complete graph with self circle
            '''
            complete graph 完全图 因为我们任意两个点都要相连！
            '''
            obj_ind = torch.arange(start, end, dtype=torch.long)# id of objects (and layout)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind)  # indexes of each vertex in the subgraph
            '''
            e.g.
            >>> obj_ind
            tensor([3, 4, 5, 6])
            >>> subj_ind_i
            tensor([[3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                    [6, 6, 6, 6]])
            >>> obj_ind_i
            tensor([[3, 4, 5, 6],
                    [3, 4, 5, 6],
                    [3, 4, 5, 6],
                    [3, 4, 5, 6]])
            明白了，这里的subj和obj相对，类似于主语和宾语的关系，是在关系中的两个人！
            因此，obj_ind_i和subj_ind_i 矩阵相同位置的一对数组成了一个关系数对！就是代表关系！
            '''
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            '''果然：！！！
            >>> rel_ind_i
            tensor([[3, 3],
                    [3, 4],
                    [3, 5],
                    [3, 6],
                    [4, 3],
                    [4, 4],
                    [4, 5],
                    [4, 6],
                    [5, 3],
                    [5, 4],
                    [5, 5],
                    [5, 6],
                    [6, 3],
                    [6, 4],
                    [6, 5],
                    [6, 6]])
            '''
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1]  # vertices connected by relation nodes should be different
            '''
            tensor([False,  True,  True,  True,  True, False,  True,  True,  True,  True,
            False,  True,  True,  True,  True, False])
            自己和自己没有关系的建立
            '''
            rel_inds.append(rel_ind_i[rel_mask_i])
            '''
            >>> rel_ind_i[rel_mask_i]
                tensor([[3, 4],
                        [3, 5],
                        [3, 6],
                        [4, 3],
                        [4, 5],
                        [4, 6],
                        [5, 3],
                        [5, 4],
                        [5, 6],
                        [6, 3],
                        [6, 4],
                        [6, 5]])
            '''
            rel_masks.append(rel_mask_i)
            obj_masks[start:end - 1] = True  # for each subgraph, first Ni vertices are objects
            #注意！这两个都是一维的信息，现在对于维度信息更加敏感和喜欢！
            lo_masks[end - 1] = True  # for each subgraph, last 1 vertex is layout, and we don't need that again

        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)
        '''
        hope to see the different but efficient use of both of them 
        因为两者信息量类似，所以这样设计的原因是什么，方便的点在于什么，希望思考？
        '''
        subj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])  # [sum(Ni + 1), sum((Ni + 1) ** 2)]
        '''
        why => bc:
        因为是Ni+1 object node, and (Ni+1)**2 relation node, so we can define the edge 
        from the object node to the relation node 
        我觉得 subj_pred_map 和 obj_pred_map 的区别是：
        还是类似前面我们说的，类似论文中的 source to relations ans destination to relations 
        Wrs and Wrd 
        '''
        obj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])
        # map from subject (an object or layout vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (rel_inds[:, 0].view(1, -1)), 1)
        # map from object (an object or layout vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].view(1, -1)), 1)
        #need more repetition
        return rel_masks.to(device), obj_masks.to(device), lo_masks.to(device), \
               obj_obj_map.to(device), subj_pred_map.to(device), obj_pred_map.to(device)

    def forward(self, output):
        maps = self._get_map(output)#need to change get_map
        '''
        what we have get ??? 
        yeah nearly there 
        '''
        if maps is None:
            return {}
        rel_masks, obj_masks, lo_masks, obj_obj_map, subj_pred_map, obj_pred_map = maps
        #the first 3 guys is like the true or false and the last 2 guys is like the i or 0 which is ready to do the matrix multiply
        '''
        can we draw a flow chart to show the process and it seems a great idea 
        what is the input of it  -> here :::: 
                input = all_output.copy()
                input['size_cls'] = data['size_cls']
                input['cls_codes'] = data['cls_codes']
                input['g_features'] = data['g_features']
                input['bdb2D_pos'] = data['bdb2D_pos']
                input['K'] = data['K']
                input['split'] = data['split']
                input['rel_pair_counts'] = data['rel_pair_counts']
                refined_output = self.output_adjust(input)
                also input contains the output from the mesh reconstruction network 
        '''
        x_obj, x_pred = self._get_object_features(output, 'obj'), self._get_object_features(output, 'rel')# obj ve rel
        '''
        此时看起来不存在房间的特征诶，并且形状也有这个感觉，确实，就是没有！！！哈哈哈下面x_lo才是房间
        '''
        x_obj, x_pred = self.obj_embedding(x_obj), self.rel_embedding(x_pred)# 2 layer mlp
        x_lo = self._get_layout_features(output)#don't need this

        x_lo = self.lo_embedding(x_lo)
        #the process to construct a graph
        x_obj_lo = [] # representation of object and layout vertices
        x_pred_objlo = [] # representation of relation vertices connecting obj/lo vertices
        rel_pair = output['rel_pair_counts']# I know yeah
        for lo_index, (start, end) in enumerate(output['split']):
            x_obj_lo.append(x_obj[start:end]) # for each subgraph, first Ni vertices are objects
            x_obj_lo.append(x_lo[lo_index:lo_index+1]) # for each subgraph, last 1 vertex is layout
            #no this hhhh
            '''
            from here we can understand that we the object and the layout are very equal, right? yes 
            '''
            x_pred_objlo.append(
                x_pred[rel_pair[lo_index]:rel_pair[lo_index + 1]].reshape(end - start, end - start, -1))
            '''
            have question about the size 
            first rel_pair[lo_index]:rel_pair[lo_index+1] is the relation between the objects  n**2 
            end-start is the number of the object, no layout here. 
            but I have  a question 就是我们的这里是所有家具连两之间的关系都存在，但是真实的就是a!=b这种类型的存在
            so the a=a type exists in the x_pred right? 
            yes! 
            '''
            # only the object hh?
            print("1 ",x_pred_objlo[-1].shape)
            x_pred_objlo[-1] = F.pad(x_pred_objlo[-1].permute(2,0,1), [0, 1, 0, 1], "constant", 0.001).permute(1,2,0)
            '''
            interesting 
            x[-1] means the last object's relationship with everyone else 
            we change the last pred_objlo ??? why 
            x_pred_objlo[-1].permute(2,0,1) and permute is like the transformation 
            F.pad(x_pred_objlo[-1].permute(2,0,1), [0, 1, 0, 1], "constant", 0.001)
            F.pad(x_pred_objlo[-1].permute(2,0,1), [0, 1, 0, 1], "constant", 0.001).permute(1,2,0)
            分布输出 print out step by step 
            is that the relation between the objects and the layout 
            I remember it just randomly initialize, maybe 
            可能这个就是传说中的那个啥，没错，用来填充的东西，是吧。我们可以打印看看其形状
            '''
            print("2 ",x_pred_objlo[-1].shape)
            x_pred_objlo[-1] = x_pred_objlo[-1].reshape((end - start + 1) ** 2, -1)
            print("3 ",x_pred_objlo[-1].shape)
            '''
            I think the dimension changes from (end-start)**2 to (end-start+1)**2 why 
            '''
        x_obj = torch.cat(x_obj_lo) # from here, for compatibility with graph-rcnn, x_obj corresponds to obj/lo vertices
        x_pred = torch.cat(x_pred_objlo)
        x_pred = x_pred[rel_masks]

        '''feature level agcn'''
        obj_feats = [x_obj]
        pred_feats = [x_pred]

        start = 0
        #GCNN update for a few steps
        for group, (gcn_collect_feat, gcn_update_feat) in enumerate(zip(self.gcn_collect_feat, self.gcn_update_feat)):
            '''
            the reason to enumerate is to you know to get collector for every group
            for each group 
            '''
            for t in range(start, start + self.feat_update_step): # maybe 0 => 4
                '''update object features'''
                # message from other objects
                source_obj = gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map, 4)# what is the args 4 for?

                # message from predicate
                source_rel_sub = gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map, 0)
                source_rel_obj = gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map, 1)
                #the difference between the subj_pre_map and the obj_pred_map, and which is the one from source to relation
                #适合的时间回去结合一下论文，会有奇效，不断复盘和调整策略
                source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3 #avg
                obj_feats.append(gcn_update_feat(obj_feats[t], source2obj_all, 0))#update is alright

                '''update predicate features'''
                #对比发现下面的这里pred_feats和obj_feats的顺序和上面的相反，说明了方向性 I guess 从后往前 yes!!!!!!!
                source_obj_sub = gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(), 2)
                source_obj_obj = gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(), 3)
                source2rel_all = (source_obj_sub + source_obj_obj) / 2
                pred_feats.append(gcn_update_feat(pred_feats[t], source2rel_all, 1))
            if self.res_group and group > 0:
                obj_feats[-1] += obj_feats[start]
                pred_feats[-1] += pred_feats[start]
            start += self.feat_update_step

        obj_feats_wolo = obj_feats[-1][obj_masks]
        '''
        obj mask is helping us the tell which of the vertices are the objects and so is the layout mask 
        '''

        # branch to predict the size
        size = self.fc1(obj_feats_wolo)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(obj_feats_wolo)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(obj_feats_wolo)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(obj_feats_wolo)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)

        obj_feats_lo = obj_feats[-1][lo_masks]

        # branch for camera parameters
        cam = self.fc_1(obj_feats_lo)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0: self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(obj_feats_lo)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]

        if self.res_output:
            size += output['size_reg_result']
            ori_reg += output['ori_reg_result']
            ori_cls += output['ori_cls_result']
            centroid_reg += output['centroid_reg_result']
            centroid_cls += output['centroid_cls_result']
            offset += output['offset_2D_result']

            pitch_reg += output['pitch_reg_result']
            pitch_cls += output['pitch_cls_result']
            roll_reg += output['roll_reg_result']
            roll_cls += output['roll_cls_result']
            lo_ori_reg += output['lo_ori_reg_result']
            lo_ori_cls += output['lo_ori_cls_result']
            lo_centroid += output['lo_centroid_result']
            lo_coeffs += output['lo_coeffs_result']

        return {'size_reg_result': size, 'ori_reg_result': ori_reg,
                'ori_cls_result': ori_cls, 'centroid_reg_result': centroid_reg,
                'centroid_cls_result': centroid_cls, 'offset_2D_result': offset,
                'pitch_reg_result': pitch_reg, 'pitch_cls_result': pitch_cls, 'roll_reg_result': roll_reg,
                'roll_cls_result': roll_cls, 'lo_ori_reg_result': lo_ori_reg, 'lo_ori_cls_result': lo_ori_cls,
                'lo_centroid_result': lo_centroid, 'lo_coeffs_result': lo_coeffs}
