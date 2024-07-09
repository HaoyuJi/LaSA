from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math
import numpy as np

from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)
        self.dropout = nn.Dropout(drop_out) #0.05

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2) #（n,head,t,c）

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values) # （n,head,t,c）,（n,head,t,c）->(n,head,c,c)
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6) #（n,head,t,c）,（n,head,c） ->(n,head,t)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) #(n,head,c,c),(n,head,t,c),(n,head,t)->(n,head,t,c)

        x = x.reshape(B, L, -1) #4 head（n,t,c）
        x = self.out_projection(x)
        x = self.dropout(x) #0.05

        return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            ) #膨胀卷积
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) #19->64
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features)) #64—>64
        
    def forward(self, feature_s, feature_t, mask):
        # Feature_s comes from space (n, t, v) feature_t comes from the previous layer of time (n, t, c)
        feature_s = feature_s.permute(0, 2, 1) #(n,v,t)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s) #(n,v,t)->(n,c,t)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t) #（n,c,c）
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t) #（n,c,c),（n,c,t）->（n,c,t）
        feature_cross = feature_cross + feature_t
        feature_cross = feature_cross.permute(0, 2, 1) #(n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask
    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer #（1,2,3,4,5,6,7,8,9）
        num_SFI_layers = len(SFI_layer) #9
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate) #0.3 dropout

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers+1, kernel_size=1) #64->10
        self.conv_t = nn.Conv1d(node, n_features, 1) #V=19->64
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in 
                range(num_layers)]) #10
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

        self.embedding_channel_change = nn.Conv2d(512, num_SFI_layers+1,1)
        self.embeddding_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, mask, joint_text_embedding):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        x = self.conv_in(x) #c=64->10
        x = x + self.embedding_channel_change(joint_text_embedding) * self.embeddding_weight
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers), 1), dim=1) #（n,10,t,v）->(n,9,t,v)+(n,1,t,v)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1) #(n,v,t)
        feature_st = self.conv_t(feature_t) #(n,v,t)->(n,64,t)

        for index, layer in enumerate(self.layers): #10 layers spatia-temporal fusion
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)

        feature_st = self.conv_out(feature_st)
        return feature_st * mask
       
class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature

    
class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones(2) * np.log(1 / 0.07))  # 2.6593

        self.in_channel = in_channel
        node = 19 if dataset == "LARA" else 25

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset) #MS-G3D
        self.STI = STI(node, n_features, n_features, n_features, n_layers, SFI_layer)
        self.joint_att = Spatial_AttLayer(n_features, n_features, 32, 64, 4)
 
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)
        self.conv_feature = nn.Conv1d(n_features,512,1)
        self.conv_feature_split = nn.Conv1d(n_features, 512, 1)

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features,512,1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
        self.ff_embedding = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, joint_text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        joint_text_embedding = self.ff_embedding(joint_text_embedding)
        joint_text_embedding = joint_text_embedding.permute(1, 0).unsqueeze(0).unsqueeze(2)

        x = self.SP(x) * mask.unsqueeze(3) #MS-G3D （n,c,t,v）

        x = self.joint_att(x, joint_text_embedding)

        feature = self.STI(x, mask, joint_text_embedding)


        
        out_cls = self.conv_cls(feature)
        out_bound = self.conv_bound(feature)
        out_feature = self.conv_feature(feature)
        out_feature_split = self.conv_feature_split(feature)
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)
                out_feature = conv_stage(feature)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound,_ = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature, out_feature_split, self.logit_scale)
        else:
            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)

            for br_stage in self.brb:
                out_bound, _ = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound)


class Spatial_AttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, qk_dim, v_dim, num_heads):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=num_heads * qk_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=num_heads * qk_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels= in_channels, out_channels= v_dim, kernel_size=1)

        z = self.conv_out = nn.Conv2d(in_channels=num_heads * v_dim, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, feature, text_feature):

        N, C, T, V = feature.size()

        k_feature = text_feature.expand(N, -1, 1, -1)
        q_feature = text_feature.expand(N, -1, 1, -1)
        v_feature = feature

        q = self.query_conv(q_feature).view(N, self.num_heads, self.qk_dim, V)
        k = self.key_conv(k_feature).view(N, self.num_heads, self.qk_dim, V)
        v = self.value_conv(v_feature)

        energy = torch.einsum('nhcu,nhcv->nhuv', q,k)
        attention = energy / (np.sqrt(self.qk_dim) * 1.0)
        attention = self.softmax(attention)

        z = torch.einsum('nctu,nhuv->nhctv', v, attention).contiguous().view(N, self.num_heads * self.v_dim, T, V)
        z = self.conv_out(z) + feature

        return z