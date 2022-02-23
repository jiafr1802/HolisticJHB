from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # resnet50
        model.global_pool = nn.Identity()
        del model.fc
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                             decoder_embedding=decoder_embedding, zsl=zsl)
    elif hasattr(model, 'head'):  # tresnet
        if hasattr(model, 'global_pool'):
            model.global_pool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                               decoder_embedding=decoder_embedding, zsl=zsl)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        '''
        what is nn.LayerNorm
        what is nn.Dropout
        '''
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        '''
        nn.MultiheadAttention

        '''
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 768->2048
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 2048->768

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]  # relation_feature
        '''
        实现了cross attention
        对应了文中2.3.2 (2) Group-decoding 中的equation(4)
        cross-attn: Gq1 ←− MultiHeadAttn(Gq,E,E) 三个参数对应Q,K,V 

        后续调用为：decoder([K,bs,768], [49,bs,768])
        tgt [K,bs,768]
        memory [49,bs,768]
        '''
        tgt = tgt + self.dropout2(tgt2)  # feature+relation_feature
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# @torch.jit.script
# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder  # K

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):  # range(K) i=>第i组（共K组）  h: bsxKxD   duplicate_pooling: KxDxg
            h_i = h[:, i, :]  # ok bsx1xD
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]  # ok 1xDxg
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)  # 第i组（共K组）bsx1xg 1组g类
            '''
            out_extrap is the result  
            '''


class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        '''
        embed_len_decoder 对应文章中的K 
        '''
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            '''
            得到每个类别的word embedding 
            图中 左上角 Group Queries部分
            -> K x D 
            torch.nn.Embedding： 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。
            torch.nn.Embedding(
                num_embeddings, – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
                embedding_dim,– 嵌入向量的维度，即用多少维来表示一个符号。
                ...)
            类似于创建了一个词袋(词典)，嗯，创建一套映射规则，从词汇（所谓词汇，在我们这里也是向量）到特定维度
            （D=embedding_dim=decoder_embedding）的向量的映射
            '''
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        '''
        TransformerDecoderLayerOptimal(768,2048,0.1)

        '''
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        '''
        nn.TransformerDecoder
        参考：https://zhuanlan.zhihu.com/p/107586681 或者Pytorch manual
        下面的代码又设计了好多其属性
        '''
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            '''
            1 对于我们的情况 
            self.decoder.duplicate_factor 就是文中的g
            '''
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            '''
            self.decoder.duplicate_pooling [K,D,1] 先创建一个这样的参数矩阵模板，下文会初始化赋值
            所以是要学习的参数，让我联想到了文中Eq(3) 中 Wk (gxD) ! 没错，而且是K个Wk，一组一个 -> K x D x g 
            '''
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)  # 正态分布随机初始化权值
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7] 正好是resnet输出 原始图像 224 x 224 【4是维度是4的意思】
            embedding_spatial = x.flatten(2).transpose(1, 2)
            '''
            x.flatten(2) -> [bs, 2048, 49] 
            x.transpose(1,2) -> [bs, 49, 2048] (符合论文图中的 wh x D , wh=7x7=49, D=2048)
            '''
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        '''
        self.decoder.embed_standart
        embed_standart = nn.Linear(initial_num_features=2048, decoder_embedding=768) 全连接层（矩阵），特征维度改变
        -> embedding_spatial_786 [bs, 49, 768] 
        '''
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)
        '''
        SHAPE: [bs,49,768]
        '''
        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:  # Ours:
            query_embed = self.decoder.query_embed.weight
            '''
            if not zsl:
                query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)

                    if embed_len_decoder > num_classes:
                        embed_len_decoder = num_classes 「K=N」
                    decoder_embedding = 768
                <==> nn.Embedding(num_classes, decoder_embedding) = [K,D]
                具体关于nn.Embedding参考上文
                query_embed.requires_grad_(False) 不参与学习和改变的过程，但是不学的话效果如何
                而weight（经过我们的尝试）相当于整个词典每个词映射结果的向量
            '''
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        '''
        before: [K,D]
        after unsqueeze: [K,1,D] 
        after expand: [K,bs,D]=[K,bs,768]==tgt SHAPE
        '''
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        '''
        before: embedding_spatial_786 SHAPE [bs,49,768]--transpose-->[49,bs,768]=[wh,bs,D]
        将bs调到中间是因为 nn.MultiheadAttention中的设定（已证实）
        我猜测这一步是完成cross attention的过程 没错！
        decoder([K,bs,768], [49,bs,768])
        ->[K,bs,D]

        温馨提示：个人认为，FF环节也在decoder中实现了
        '''
        h = h.transpose(0, 1)  # -> [bs,K,D]

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        '''
        为什么要创造纯0 out_extrap 
        明白了！这个其实是下面一行 self.decoder.group_fc 会把其运算结果放到out_extrap中
        这里创造是等待写入内容
        '''
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        '''
        group fully-connected pooling 

        '''
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1) # [bs,N] (N=num class)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        return logits
