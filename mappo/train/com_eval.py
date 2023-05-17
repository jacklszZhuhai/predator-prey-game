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

def check(input):
    output = th.from_numpy(input) if type(input) == np.ndarray else input
    return output

def _t2n(x):
    return x.detach().cpu().numpy()

def pybullet_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = PybulletEnv(all_args)
            # env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.nr)])


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


'''class MergeLayer(nn.Module):
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
        return self.layer_norm_2(self.act(self.fc2(h)))'''
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
        #h = self.layer_norm(h)
        h = self.act(self.fc2(h))
        return h

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


'''class cons_kd(nn.Module):
    def __init__(self, obs_dim, d_model, agent_num):
        super(cons_kd, self).__init__()
        self.merge_layer = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=d_model)
        #self.att = qkv_net(d_model=d_model, agent_num=agent_num, head=4)
        self.att = Simple_att(hid=d_model, agent_num=agent_num)
        self.att = att_net(d_model=d_model, nhead=8, layer=4, agent_num=agent_num)

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
        return m_new'''


class cons_kd(nn.Module):
    def __init__(self, obs_dim, d_model, agent_num):
        super(cons_kd, self).__init__()
        self.merge_layer = MergeLayer(in1=obs_dim, in2=d_model, hid=d_model, out=d_model)
        # self.att = qkv_net(d_model=d_model, agent_num=agent_num, head=4)
        # self.att = Simple_att(hid=d_model, agent_num=agent_num)
        self.att = att_net(d_model=d_model, nhead=4, layer=2, agent_num=agent_num)

        self.resnet1 = nn.Linear(obs_dim, d_model)
        self.resnet2 = nn.Linear(2 * d_model, d_model)
        nn.init.orthogonal_(self.resnet1.weight)
        nn.init.orthogonal_(self.resnet2.weight)

    def forward(self, obs, last_hid, dis):
        # obs [batch, obs_dim]  dis [batch, agent_num]
        # last_hid [batch, agent_num, d_model]
        b, n, d = last_hid.size()
        index = th.argmin(dis, dim=1)
        temp = th.zeros([b, d])  # temp [batch, d_model]
        if last_hid.is_cuda:
            temp = temp.cuda()
        t_h = deepcopy(last_hid)  # t_h [batch, agent_num, d_model]
        for i in range(b):
            temp[i] = last_hid[i, index[i], :]
        m_curr = self.merge_layer(obs, temp)  # [agent_num(batch), d_model]
        # m_curr = self.gru(obs, last_h)
        for i in range(b):
            t_h[i, index[i], :] = m_curr[i]
        m_hidden = t_h  # [agent_num(batch), agent_num, d_model]
        # m_curr = m_curr.unsqueeze(dim=1)
        # m_new, _ = self.att(src=m_curr, hidden=last_hid, dis=dis)
        m_new, _ = self.att(hidden=m_hidden, dis=dis)

        obs_hidden = nn.ReLU()(self.resnet1(obs))
        x = th.cat([obs_hidden, m_new], dim=1)
        h = nn.ReLU()(self.resnet2(x))

        return m_new

class Actor(nn.Module):
    def __init__(self, hidden_size, num_agents, cl, n_r):
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


class Cons_net:
    def __init__(self, args):
        self.read_dir = args.read_dir
        self.batch_size = args.batch_size
        self.steps = args.steps
        self.num = args.num_agents
        self.d_model = args.d_model
        self.save_dir = args.save_dir
        self.ori_dir = args.ori_dir
        self.lr = args.lr
        self.loss_type = args.loss_fun
        self.envs = pybullet_train_env(args)

        self.obs_dim = self.num * 6
        self.cl = args.comm_dis
        self.nr = args.nr
        self.test_cons = args.test_cons
        self.kd_network = cons_kd(obs_dim=self.obs_dim,
                                  d_model=self.d_model, agent_num=self.num)
        self.actor = Actor(self.d_model, self.num, self.cl, self.nr)
        self.actor_kdpart = Actor_part(hidden_size=self.d_model, num_agents=self.num)
        self.hid_states = np.zeros((self.steps + 1, self.nr, self.num,
                                    self.num, self.d_model),dtype=np.float32)
        self.obs = np.zeros((self.steps + 1, self.nr, self.num,
                             self.obs_dim + self.num),dtype=np.float32)
        self.actions = np.zeros((self.steps, self.nr, self.num,
                                 2), dtype=np.float32)
        self.rewards = np.zeros((self.steps, self.nr, self.num,
                                 1), dtype=np.float32)

    def eval(self):
        self.kd_network.eval()
        self.actor_kdpart.eval()
        self.actor.eval()
        for eval_step in range(20):
            self.warmup()
            eval_episode_rewards = []
            for step in range(self.steps):
                # Sample actions
                obs_temp = check(np.concatenate(self.obs[step]))
                hid_temp = check(np.concatenate(self.hid_states[step]))
                obs_dis = obs_temp[:, -self.num:]
                obs_in = obs_temp[:, :-self.num]
                if self.test_cons and eval_step<10:# and step%5:
                    # dis_in = obs_dis.clip(0, self.cl)
                    actor_features = self.kd_network(obs=obs_in,
                                                       last_hid=hid_temp,
                                                       dis=obs_dis)
                    hid_s_ori = actor_features.reshape(self.nr, self.num, -1)
                    hid_s_t = hid_s_ori.repeat(1, self.num, 1) \
                        .reshape(self.nr, self.num, self.num, self.d_model)
                    index_dis = obs_dis.argsort(dim=1).reshape(self.nr, self.num, -1)
                    att_dis = obs_dis.reshape(self.nr, self.num, -1)
                    for i in range(self.nr):
                        hid_temp2 = hid_s_ori[i]
                        for agent in range(self.num):
                            inx_temp = index_dis[i, agent, :]
                            for a in range(self.num):
                                if att_dis[i, agent, inx_temp[a]] < self.cl or a < 2:  # < self._n_agent/2:
                                    hid_s_t[i, agent, inx_temp[a], :] = hid_temp2[inx_temp[a]]
                                else:
                                    hid_s_t[i, agent, inx_temp[a], :] = 0  # torch.zeros(self._att_hidden)
                    hid = hid_s_t.reshape(self.nr * self.num, self.num, self.d_model)
                    actions_env = self.actor.get_action(actor_features)
                else:
                    actions_env, hid = self.actor(obs=obs_in)
                    actor_features = self.actor_kdpart(obs=obs_in)
                    actions_env = self.actor.get_action(actor_features)
                #actions_env= self.kd_network(self.obs)
                # Obser reward and next obs
                # print(actions_env)
                actions = np.array(np.split(_t2n(actions_env), self.nr))
                hid_n = np.array(np.split(_t2n(hid), self.nr))
                obs, rewards, dones, infos = self.envs.step(actions)
                eval_episode_rewards.append(rewards)
                self.obs[step+1] = obs[:, 1:, :]
                self.hid_states[step+1] = hid_n
                self.actions[step] = actions
                self.rewards[step] = rewards

            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_average_episode_rewards = np.mean(np.sum(np.array(eval_episode_rewards), axis=0))
            print("eval average episode rewards of agent:",eval_average_episode_rewards)

    def train(self):
        self.cons_optimizer = th.optim.Adam(self.kd_network.parameters(), lr=1e-5)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.actor.eval()
        self.actor_kdpart.eval()
        for eval_step in range(400):
            self.warmup()
            eval_episode_rewards = []
            loss_a = 0
            loss_a2 = 0
            for step in range(self.steps):
                # Sample actions
                obs_temp = check(np.concatenate(self.obs[step]))
                hid_temp = check(np.concatenate(self.hid_states[step]))
                obs_dis = obs_temp[:, -self.num:]
                obs_in = obs_temp[:, :-self.num]

                self.cons_optimizer.zero_grad()
                actor_features = self.kd_network(obs=obs_in,
                                                   last_hid=hid_temp,
                                                   dis=obs_dis)
                base = self.actor.get_fea(obs=obs_in).detach()

                loss = self.loss(F.log_softmax(actor_features, dim=1), F.softmax(base, dim=1))
                loss.backward()
                self.cons_optimizer.step()

                af_temp = deepcopy(actor_features.detach())
                hid_s_ori = af_temp.reshape(self.nr, self.num, -1)
                hid_s_t = hid_s_ori.repeat(1, self.num, 1) \
                    .reshape(self.nr, self.num, self.num, self.d_model)
                index_dis = obs_dis.argsort(dim=1).reshape(self.nr, self.num, -1)
                att_dis = obs_dis.reshape(self.nr, self.num, -1)
                for i in range(self.nr):
                    hid_temp2 = hid_s_ori[i]
                    for agent in range(self.num):
                        inx_temp = index_dis[i, agent, :]
                        for a in range(self.num):
                            if att_dis[i, agent, inx_temp[a]] < self.cl or a < 2:  # < self._n_agent/2:
                                hid_s_t[i, agent, inx_temp[a], :] = hid_temp2[inx_temp[a]]
                            else:
                                hid_s_t[i, agent, inx_temp[a], :] = -1e9  # torch.zeros(self._att_hidden)
                hid = hid_s_t.reshape(self.nr * self.num, self.num, self.d_model)
                actions_env = self.actor.get_action(af_temp)
                actions = np.array(np.split(_t2n(actions_env), self.nr))
                hid_n = np.array(np.split(_t2n(hid), self.nr))
                obs, rewards, dones, infos = self.envs.step(actions)
                eval_episode_rewards.append(rewards)
                self.obs[step+1] = obs[:, 1:, :]
                self.hid_states[step+1] = hid_n
                self.actions[step] = actions
                self.rewards[step] = rewards
                loss_a += np.array(loss.data)
                tes = self.actor_kdpart(obs_in).detach()
                loss_2 = self.loss(F.log_softmax(tes, dim=1), F.softmax(base, dim=1))
                loss_a2 += np.array(loss_2.data)
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_average_episode_rewards = np.mean(np.sum(np.array(eval_episode_rewards), axis=0))
            print("rewards:",eval_average_episode_rewards,
                  "act_loss",loss_a/self.steps,
                  "kdact_loss",loss_a2/self.steps)

            self.save_model()

    def save_model(self):
        """Save networks."""
        #print(str(self.save_dir))
        th.save(self.kd_network.state_dict(), str(self.save_dir) + "/cons_kd_ft.pt")

    def warmup(self):
        # reset env
        obs_get = self.envs.reset()  # shape = (5, 2, 14)       3,20
        obs = obs_get[:, 1:, :]
        self.obs[0] = obs.copy()

    def restore(self, pre=None):
        """Restore policy's networks from a saved model."""
        if pre is None:
            dir = self.save_dir
        else:
            dir = pre
        self.model_dir = dir
        policy_cons_state_dict = th.load(str(self.model_dir) + '/cons_kd_cl3_oriresnet_n.pt', map_location=th.device('cpu'))
        self.kd_network.load_state_dict(policy_cons_state_dict)

        policy_part_state_dict = th.load(str(self.model_dir) + '/act_kd_cl3.pt', map_location=th.device('cpu'))
        self.actor_kdpart.load_state_dict(policy_part_state_dict)

        policy_actor_state_dict = th.load(str(self.ori_dir) + '/actor.pt', map_location=th.device('cpu'))
        self.actor.load_state_dict(policy_actor_state_dict)


if __name__ == "__main__":
    re_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                   + "/results/ATTN/formation/mappo/noattn_a5_n8_hd/run1/models")
    ori_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                   + "/results/ATTN/formation/mappo/noattn_a5_n8_hd/run1/models")
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default=re_dir)
    parser.add_argument('--save_dir', type=str, default=re_dir)
    parser.add_argument('--ori_dir', type=str, default=ori_dir)
    parser.add_argument('--test_cons', type=str, default=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--episodes', type=int, default=100000)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--num_agents', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--loss_fun', type=str, default='MSE')

    parser.add_argument('--nr', type=int, default=1)
    parser.add_argument('--comm_dis', type=float, default=3)

    all_args = parser.parse_known_args()[0]
    net = Cons_net(all_args)
    net.restore()
    net.eval()
    #net.train()



