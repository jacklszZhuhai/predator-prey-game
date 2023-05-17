import torch
import torch as th
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .util import init, get_clones
from mappo.utils.util import get_shape_from_obs_space


class MergeLayer(nn.Module):
    def __init__(self, in1, in2, hid, out):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = nn.Linear(in1 + in2, hid)
        self.fc2 = nn.Linear(hid, out)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hid)
        self.layer_norm_2 = nn.LayerNorm(out)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        # h = self.layer_norm(h)
        h = self.act(self.fc2(h))
        # h = self.layer_norm_2(h)
        return h


class Merge_Single(nn.Module):
    def __init__(self, in1, in2, out):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = nn.Linear(in1 + in2, out)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out)
        nn.init.orthogonal_(self.fc1.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)
        h = self.layer_norm(self.act(self.fc1(x)))
        return h


class Simple_att(nn.Module):
    def __init__(self, hid, agent_num):
        super().__init__()
        self.e_dim = int(hid / 4)
        self.emb = DisEncode(expand_dim=self.e_dim)
        self.merge_layer = MergeLayer(in1=hid, in2=hid, hid=hid, out=hid)
        self.fc1 = nn.Linear((hid + self.e_dim) * agent_num, (hid + self.e_dim) * agent_num)
        self.fc2 = nn.Linear((hid + self.e_dim) * agent_num, hid)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm((hid + self.e_dim) * agent_num)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, src, hidden, dis):
        # src [batch, 1, d_model]  hidden [batch, agent_num, d_model]  dis [batch, agent_num]
        dis_embed = self.emb(dis)  # [batch, agent_num, e_dim]
        x = th.cat([hidden, dis_embed], dim=2)  # [batch, agent_num, e_dim+d_model]
        # x = self.merge_layer(hidden, dis_embed)
        h = self.layer_norm(self.act(self.fc1(x.flatten(1))))
        h = self.fc2(h)  # [batch, d_model]
        out = self.merge_layer(src.squeeze(), h)  # [batch, d_model]
        return out, h


class DisEncode(nn.Module):
    def __init__(self, expand_dim):
        super(DisEncode, self).__init__()
        d_model = expand_dim
        # basis_freq [1, d_model]
        self.w = nn.Linear(1, d_model)
        self.w.weight = nn.Parameter((th.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float().reshape(d_model, -1))
        self.w.bias = nn.Parameter(th.zeros(d_model).float())

    def forward(self, dis):
        # dis [batch, agent_num]
        batch_size = dis.size(0)
        seq_len = dis.size(1)
        dis = dis.view(batch_size, seq_len, 1)  # [batch, agent_num, 1]
        harmonic = th.cos(self.w(dis))  # [batch, agent_num, d_model]
        return harmonic

class qkvnet(nn.Module):
    def __init__(self, d_model=128, agent_num=5, head=2):
        super(qkvnet, self).__init__()
        self.d_model = d_model
        self.e_dim = int(d_model/2)
        self.emb = DisEncode(expand_dim=self.e_dim)
        self.query_dim = d_model + self.e_dim
        self.key_dim = d_model + self.e_dim
        self.merge_layer = MergeLayer(in1=self.query_dim, in2=self.query_dim, hid=d_model,out=d_model)
        self.fc1 = nn.Linear(self.query_dim, d_model)
        nn.init.orthogonal_(self.fc1.weight)
        self.act = nn.ReLU()
        self.m_a = nn.MultiheadAttention(embed_dim=self.query_dim, 
                                        kdim=self.key_dim,
                                        vdim=self.key_dim,
                                        num_heads=head,
                                        dropout=0.1,
                                        batch_first=True)

    def forward(self, hidden, dis):
        # hidden [batch(agent_num), agent_num, d_model]  dis [batch(agent_num), agent_num]
        dis_embed = self.emb(dis)
        input_src = th.cat([hidden, dis_embed], dim=2)
        src = input_src[:, 0, :].unsqueeze(dim=1)
        attn_output, _ =self.m_a(src, input_src[:,1:,:], input_src[:,1:,:])
        attn_output = attn_output.squeeze()
        #out = self.merge_layer(src.squeeze(), attn_output)
        out = attn_output + src.squeeze()
        out = self.act(self.fc1(out))
        return out, attn_output
	
################# transformer注意力层
# from .trans import make_denoiser1
class att_net(nn.Module):
    def __init__(self, d_model=128, nhead=4, layer=2, agent_num=5):
        super(att_net, self).__init__()
        self.d_model = d_model
        self.emb = DisEncode(expand_dim=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=nhead, dim_feedforward=2048,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layer)
        # self.trans = make_denoiser1()
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=2*d_model, nhead=nhead)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layer)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        self.merge_layer = MergeLayer(in1=agent_num * 2 * d_model, in2=agent_num * 2 * d_model, hid=2 * d_model,
                                      out=d_model)

    def forward(self, hidden, dis):
        # hidden [batch(agent_num), agent_num, d_model]  dis [batch(agent_num), agent_num]
        dis_embed = self.emb(dis)  # [batch(agent_num), agent_num, d_model]
        input_src = th.cat([hidden, dis_embed], dim=2)  # [batch, agent_num, 2*d_model]
        # input_src_t = input_src.permute(1, 0, 2)  # [agent_num, batch, 2*d_model]############
        # input_mask = mask.unsqueeze(dim=2).expand(-1, -1, mask.size()[1]).permute(1, 0, 2)
        encode = self.transformer_encoder(src=input_src)  # [agent_num, batch, 2*d_model]
        # encode = self.trans(input_src)
        # encode = encode.permute(1, 0, 2)  # [batch, agent_num, 2*d_model]
        out = self.merge_layer(input_src.flatten(start_dim=1), encode.flatten(start_dim=1))  # [batch, d_model]
        return out, encode

################# 最主要的架构
class new_cons(nn.Module):
    def __init__(self, obs_dim=30, d_model=128, agent_num=5, out_d=128):
        super().__init__()
        # self.gru = nn.GRU(obs_dim, d_model, num_layers=1)
        self.merge_layer = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=d_model)
        self.att = att_net(d_model=d_model, nhead=4, layer=2, agent_num=agent_num)
        # self.resnet = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=out_d)
        #self.att = qkvnet(d_model=d_model, agent_num=agent_num, head=4)
        # self.att = Simple_att(hid=d_model, agent_num=agent_num)
        # self.fc1 = nn.Linear(d_model, d_model)
        # self.fc2 = nn.Linear(d_model, whole_dim)
        # self.mlp = MLPBase(args, obs_dim)
        self.estimate_layer = Estimate(d_model=out_d, out_est=obs_dim)

    def forward(self, obs, last_hid, dis):
        # obs [batch, obs_dim]  dis [batch, agent_num]
        # last_hid [batch, agent_num, d_model]
        # num = self.agent_num
        ##############################
        b, n, d = last_hid.size()
        #index = torch.argmin(dis, dim=1)
        temp = torch.zeros([b, d])  # temp [batch, d_model]
        if obs.is_cuda:
            temp = temp.cuda()
        t_h = deepcopy(last_hid)  # t_h [batch, agent_num, d_model]
        #for i in range(b):
        #    temp[i] = last_hid[i, index[i], :]
        temp[:,:] = last_hid[:, 0, :]
        m_curr = self.merge_layer(obs, temp)  # ([batch, obs_dim], [batch, d_model])    # 将观测obs_dim和last_hid进行拼接合并
        # m_curr = self.gru(obs, last_h)
        #for i in range(b):
        #    t_h[i, index[i], :] = m_curr[i]
        t_h[:, 0, :] = m_curr[:]                    # 将last_hid自己对应的那一列用m_curr替换
        m_hidden = t_h  # [batch, agent_num, d_model]
        # m_hidden = th.cat([m_curr.unsqueeze(dim=1), last_hid], dim=1)  # [batch, agent_num+1, d_model]
        # m_hidden = th.cat([m_curr.unsqueeze(dim=1), last_hid[:, 1:, :]], dim=1)  # [batch, agent_num, d_model]
        ##########################################
        # for i in range(num):
        #    m_hidden[i, i, :] = m_curr[i, :]
        # m_curr = m_curr.unsqueeze(dim=1)
        # m_new, _ = self.att(src=m_curr, hidden=last_hid, dis=dis)
        m_new, _ = self.att(hidden=m_hidden, dis=dis)
        # m_new [batch(agent_num), d_model]  attn [batch, agent_num, 2*d_model]
        # obs_hidden = nn.ReLU()(self.resnet1(obs))
        # x = th.cat([obs_hidden, m_new], dim=1)
        # h = nn.ReLU()(self.resnet2(obs_hidden)) + m_new
        # h = self.layer_norm(h)
        estimate = self.estimate_layer(m_new)
        # estimate [batch, obs_dim]
        return m_new, estimate


class Res(nn.Module):
    def __init__(self, obs_dim=30, d_model=128, out_d=128):
        super().__init__()
        self.resnet1 = nn.Linear(obs_dim, d_model)
        self.resnet2 = nn.Linear(d_model, out_d)
        self.wei = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(out_d)
        nn.init.orthogonal_(self.resnet1.weight)
        nn.init.orthogonal_(self.resnet2.weight)
        nn.init.orthogonal_(self.wei.weight)

    def forward(self, obs):
        obs_hidden = nn.ReLU()(self.resnet1(obs))
        # x = th.cat([obs_hidden, m_new], dim=1)
        #h = nn.ReLU()(self.resnet2(obs_hidden))
        #h = self.layer_norm(h)
        w = nn.Tanh()(self.wei(obs_hidden))
        w = 0.5 * w + 0.5

        return w

################# 估计全局信息的网络
class Estimate(nn.Module):
    def __init__(self, d_model=128, out_est=30):
        super().__init__()
        self.dim = d_model
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, out_est)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.fc3 = nn.Linear(2 * d_model, whole_dim)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        # nn.init.orthogonal_(self.fc3.weight)

    def forward(self, m_new):
        # m_new [batch, d_model]
        # m_new = m_new.reshape(-1, self.dim)
        est = self.fc2(nn.ReLU()(self.fc1(m_new)))
        est = nn.Tanh()(est)
        # est = self.fc3(nn.ReLU()(est))
        # estimate [batch, out_est]
        return est
