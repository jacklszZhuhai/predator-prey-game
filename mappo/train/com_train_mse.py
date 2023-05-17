import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from mappo.algorithms.utils.trans import MultiHeadedAttention
from mappo.utils.memory import ReplayMemory
import argparse
import os
from pathlib import Path
from mappo.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from mappo.env import PybulletEnv
from mappo.algorithms.utils.act import ACTLayer
from mappo.algorithms.utils.mlp import MLPSimple
from gym import spaces

class Actor(nn.Module):
    def __init__(self, hidden_size, num_agents, cl=20, n_r=1):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self._n_agent = num_agents
        self._n_roll = n_r
        self.cl = cl
        self.base = MLPSimple(self._n_agent*6, self.hidden_size, 2)
        a_s = spaces.Box(low=-1., high=1., shape=(2,),
                                                     dtype=np.float32)
        self.act = ACTLayer(a_s, self.hidden_size, True, 0.01)

    def forward(self, obs, available_actions=None, deterministic=False):
        actor_features = self.base(obs)
        hid_s_ori = actor_features.reshape(self._n_roll, self._n_agent, -1)
        hid_s_t = hid_s_ori.repeat(1, self._n_agent, 1) \
            .reshape(self._n_roll, self._n_agent, self._n_agent, self.hidden_size)
        hid_state = hid_s_t.reshape(self._n_roll * self._n_agent, self._n_agent, self.hidden_size)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, hid_state

    def get_action(self, actor_features, available_actions=None, deterministic=False):
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions

    def get_fea(self, obs):
        actor_features = self.base(obs)
        return actor_features


class Actor_part(nn.Module):
    def __init__(self, hidden_size, num_agents):
        super(Actor_part, self).__init__()
        self.hidden_size = hidden_size
        self._n_agent = num_agents
        self.base = MLPSimple(self._n_agent*6, self.hidden_size, 2)

    def forward(self, obs):
        actor_features = self.base(obs)
        return actor_features


class qkv(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hid)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=1)
        h = self.layer_norm(self.act(self.fc1(x)))
        return self.fc2(h)


class qkv_net(nn.Module):
    def __init__(self, d_model=128, agent_num=5, head=2):
        super(qkv_net, self).__init__()
        self.d_model = d_model
        self.e_dim = int(d_model / 4)
        self.emb = DisEncode(expand_dim=self.e_dim)
        #self.q = qkv(d_model, d_model)
        #self.k = qkv(d_model, d_model)
        #self.v = qkv(d_model, d_model)
        self.query_dim = self.d_model
        self.key_dim = (self.d_model + self.e_dim)
        self.merge_layer = MergeLayer(in1=self.query_dim,in2=self.query_dim,hid=d_model,out=d_model)
        #self.attn =MultiHeadedAttention(h=head, d_model=d_model*2)
        self.m_a = nn.MultiheadAttention(embed_dim=self.query_dim,
                                         kdim=self.key_dim,
                                         vdim=self.key_dim,
                                         num_heads=head,
                                         dropout=0.1,
                                         batch_first=True)

    def forward(self, src, hidden, dis):
        # src [batch, 1, d_model]  hidden [batch, agent_num, d_model]  dis [batch, agent_num]
        dis_embed = self.emb(dis)  # [batch, agent_num, e_dim]
        input_src = th.cat([hidden, dis_embed], dim=2)  # [batch, agent_num, query_dim]
        # input_mask = mask.unsqueeze(dim=2).expand(-1, -1, mask.size()[1]).permute(1, 0, 2)
        #attn_output, attn_output_weights
        attn_output, attn_output_weights = self.m_a(src, input_src, input_src)  # [batch, 1, d_model]
        attn_output = attn_output.squeeze()
        out = self.merge_layer(src.squeeze(), attn_output)  # [batch, d_model]
        return out, attn_output


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
        h = self.layer_norm(self.act(self.fc1(x)))
        return self.layer_norm_2(self.act(self.fc2(h)))

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


class Simple_att(nn.Module):
    def __init__(self, hid, agent_num):
        super().__init__()
        self.e_dim = int(hid / 4)
        self.emb = DisEncode(expand_dim=self.e_dim)
        self.merge_layer = MergeLayer(in1=hid,in2=hid,hid=hid,out=hid)
        self.fc1 = nn.Linear((hid + self.e_dim)*agent_num, (hid + self.e_dim)*agent_num)
        self.fc2 = nn.Linear((hid + self.e_dim)*agent_num, hid)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm((hid + self.e_dim)*agent_num)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, src, hidden, dis):
        # src [batch, 1, d_model]  hidden [batch, agent_num, d_model]  dis [batch, agent_num]
        dis_embed = self.emb(dis)  # [batch, agent_num, e_dim]
        x = th.cat([hidden, dis_embed], dim=2)  # [batch, agent_num, e_dim+d_model]
        #x = self.merge_layer(hidden, dis_embed)
        h = self.layer_norm(self.act(self.fc1(x.flatten(1))))
        h = self.fc2(h)  # [batch, d_model]
        out = self.merge_layer(src.squeeze(), h)  # [batch, d_model]
        return out, h

class att_net(nn.Module):
    def __init__(self, d_model=128, nhead=4, layer=2, agent_num=5):
        super(att_net, self).__init__()
        self.d_model = d_model
        self.emb = DisEncode(expand_dim=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2*d_model, nhead=nhead,
                                                        dim_feedforward=1024, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layer)
        self.merge_layer = MergeLayer(in1=agent_num*2*d_model, in2=agent_num*2*d_model, hid=2*d_model, out=d_model)

    def forward(self, hidden, dis):
        # hidden [batch(agent_num), agent_num, d_model]  dis [batch(agent_num), agent_num]
        dis_embed = self.emb(dis)  # [batch(agent_num), agent_num, d_model]
        input_src = th.cat([hidden, dis_embed], dim=2)  # [batch, agent_num, 2*d_model]
        encode = self.transformer_encoder(src=input_src)  # [agent_num, batch, 2*d_model]
        out = self.merge_layer(input_src.flatten(start_dim=1), encode.flatten(start_dim=1))  # [batch, d_model]
        return out, encode

class cons_kd(nn.Module):
    def __init__(self, obs_dim, d_model, agent_num):
        super(cons_kd, self).__init__()
        self.merge_layer = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=d_model)
        #self.att = qkv_net(d_model=d_model, agent_num=agent_num, head=4)
        #self.att = Simple_att(hid=d_model, agent_num=agent_num)
        self.att = att_net(d_model=d_model, nhead=4, layer=2, agent_num=agent_num)

    def forward(self, obs, last_hid, dis):
        # obs [batch, obs_dim]  dis [batch, agent_num]
        # last_hid [batch, agent_num, d_model]
        b, n, d = last_hid.size()
        index = th.argmin(dis, dim=1)
        temp = th.zeros([b, d])  # temp [batch, d_model]
        t_h = deepcopy(last_hid)  # t_h [batch, agent_num, d_model]
        for i in range(b):
            temp[i] = last_hid[i, index[i], :]
        m_curr = self.merge_layer(obs, temp)  # [agent_num(batch), d_model]
        # m_curr = self.gru(obs, last_h)
        for i in range(b):
            t_h[i, index[i], :] = m_curr[i]
        m_hidden = t_h  # [agent_num(batch), agent_num, d_model]
        #m_curr = m_curr.unsqueeze(dim=1)
        #m_new, _ = self.att(src=m_curr, hidden=last_hid, dis=dis)
        m_new, _ = self.att(hidden=m_hidden, dis=dis)
        return m_new


class Cons_net:
    def __init__(self, args):
        self.read_dir = args.read_dir
        self.batch_size = args.batch_size
        self.episodes = args.episodes
        self.num = args.num_agents
        self.d_model = args.d_model
        self.save_dir = args.save_dir
        self.ori_dir = args.ori_dir
        self.lr = args.lr
        self.loss_type = args.loss_fun
        self.comm_dis = args.comm_dis
        self.obs_dim = self.num * 6
        self.kd_network = cons_kd(obs_dim=self.obs_dim, d_model=self.d_model, agent_num=self.num)
        self.actor = Actor_part(hidden_size=self.d_model,num_agents=self.num)
        self.base = Actor(hidden_size=self.d_model, num_agents=self.num)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.cons_optimizer = th.optim.Adam(self.kd_network.parameters(), lr=self.lr)
        lambda1 = lambda epoch: 0.1 ** (epoch)
        self.sch = th.optim.lr_scheduler.LambdaLR(self.cons_optimizer, lr_lambda=lambda1)
        self.sch2 = th.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda1)
        self.memory = ReplayMemory(capacity=100000, save_dir=self.read_dir, new=False)
        self.use_cuda = th.cuda.is_available()
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()
        #self.loss = nn.KLDivLoss(reduction='batchmean')

    def train(self):
        self.load_base()
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        loss_a = 0
        loss_ae = 0
        print(str(self.save_dir))
        if self.comm_dis < 20:
            self.memory.comm_limit(self.comm_dis, self.num)
        for episode in range(self.episodes):
            batch = self.memory.sample(self.batch_size)
            hidden_batch = th.tensor(batch.hidden).type(FloatTensor)
            hidden_next_batch = th.tensor(batch.hidden_next).type(FloatTensor)
            obs_batch = th.tensor(batch.obs_total).type(FloatTensor)

            if self.comm_dis < 20:
                obs_batch_kd = obs_batch[:, 0, :]
                obs_batch_base = obs_batch[:, 1, :]
            else:
                obs_batch_kd = obs_batch
                obs_batch_base = obs_batch
            dis_kd = obs_batch_kd[:, -self.num:]
            obs_kd = obs_batch_kd[:, :-self.num]

            obs_base = obs_batch_base[:, :-self.num]
            base = self.base.get_fea(obs=obs_base).detach()

            est_ac = self.actor(obs=obs_kd)
            self.actor_optimizer.zero_grad()
            loss = self.loss(est_ac, base)
            #loss = self.loss(F.log_softmax(est_ac, dim=1), F.softmax(base, dim=1))
            loss.backward()
            #loss2 = self.loss(F.log_softmax(base, dim=1), F.softmax(hidden_next_batch, dim=1))
            self.actor_optimizer.step()

            self.cons_optimizer.zero_grad()
            est = self.kd_network(obs=obs_kd, last_hid=hidden_batch, dis=dis_kd)
            loss_e = self.loss(est, base)
            #loss_e = self.loss(F.log_softmax(est, dim=1), F.softmax(base, dim=1))
            loss_e.backward()
            self.cons_optimizer.step()

            loss_a += np.array(loss.data)
            loss_ae += np.array(loss_e.data)
            if episode % 100 == 0 and episode>0:
                print('epoch:',episode,"act_loss",loss_a/100, "cons_loss", loss_ae / 100)
                #if episode%10000==0 and episode>30000:
                #if episode == 20000 :
                #    print('lr:', self.sch.get_last_lr())
                #    self.sch.step()
                #    self.sch2.step()
                self.save_model()
                if loss_a/100 < 1e-5 and loss_ae/100 < 1e-5:
                    break
                loss_a = 0
                loss_ae = 0

    def save_model(self):
        """Save networks."""
        #print(str(self.save_dir))
        th.save(self.kd_network.state_dict(), str(self.save_dir) + "/cons_kd_cl3_mse.pt")
        th.save(self.actor.state_dict(), str(self.save_dir) + "/act_kd_cl3_mse.pt")

    def load_base(self):
        policy_actor_state_dict = th.load(str(self.ori_dir) + '/actor.pt', map_location=th.device('cpu'))
        self.base.load_state_dict(policy_actor_state_dict)
        self.base.eval()

        #policy_cons_state_dict = th.load(str(self.save_dir) + '/cons_kd_cl3.pt', map_location=th.device('cpu'))
        #self.kd_network.load_state_dict(policy_cons_state_dict)

        #policy_part_state_dict = th.load(str(self.save_dir) + '/act_kd_cl3.pt', map_location=th.device('cpu'))
        #self.actor.load_state_dict(policy_part_state_dict)


if __name__ == "__main__":
    re_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                   + "/results/ATTN/formation/mappo/noattn_a5_n8_hd/run1/models")
    ori_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                   + "/results/ATTN/formation/mappo/noattn_a5_n8_hd/run1/models")
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default=re_dir)
    parser.add_argument('--save_dir', type=str, default=re_dir)
    parser.add_argument('--ori_dir', type=str, default=ori_dir)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--episodes', type=int, default=500000)
    parser.add_argument('--num_agents', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--loss_fun', type=str, default='MSE')
    parser.add_argument('--comm_dis', type=float, default=3)

    all_args = parser.parse_known_args()[0]
    net = Cons_net(all_args)
    net.train()

