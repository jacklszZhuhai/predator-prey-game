"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""

import torch
import torch.nn as nn
from mappo.algorithms.utils.util import init, check
from mappo.algorithms.utils.cnn import CNNBase
from mappo.algorithms.utils.mlp import MLPBase
from mappo.algorithms.utils.rnn import RNNLayer
from mappo.algorithms.utils.act import ACTLayer
from mappo.algorithms.utils.att import new_cons, Res
from mappo.algorithms.utils.popart import PopArt
from mappo.utils.util import get_shape_from_obs_space
from copy import deepcopy

########### Actor网络架构
class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), num=0):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_attn = args.use_attn
        self._att_hidden = args.att_hidden
        self._n_agent = num
        self._n_roll = args.n_rollout_threads
        self.step = args.episode_length
        self.cl = args.comm_dis
        self.count = 0
        obs_shape = get_shape_from_obs_space(obs_space)
        ################ 是否使用attention架构
        if self._use_attn:
            print('use attn')
            obs_dim = obs_shape[0]
            ############ attention模块
            self.base = new_cons(obs_dim=obs_dim, d_model=self._att_hidden,
                                 agent_num=self._n_agent, out_d=self.hidden_size)
            ############ 之前是resnet，现在只用来计算attention模块与MLP模块的加权的权重w
            self.resnet = Res(obs_dim=obs_dim, d_model=self._att_hidden, out_d=self.hidden_size)
            ############ MLP组成的resnet，层数和隐藏层数量都由args决定
            self.resnet_mlp = MLPBase(args, obs_shape)
            ############ 下面这些参数在好几个版本前用过，但现在改完代码都不需要了还没删
            # for eval
            self.dis = torch.zeros([self.step, self._n_roll, self._n_agent, self._n_agent])
            self.last_hid = torch.zeros([self.step + 1, self._n_roll, self._n_agent, self.hidden_size])

            # for train
            self.last_hid_tr = torch.zeros([self.step, self._n_roll, self._n_agent, self.hidden_size])
            self.curr_hid_tr = torch.zeros([self.step, self._n_roll, self._n_agent, self.hidden_size])
            self.est_temp_tr = torch.zeros([self.step, self._n_roll, self._n_agent, obs_dim])
            
        else:
            ########### 不使用attention模块时普通的MLP网络，层数和隐藏层数量都由args决定
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)
        ##### 是否用RNN，目前没用
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        ##### actor的隐藏层转动作输出的网络
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)
    
    ######### 这个函数之前用过，现在没什么用
    def init_att(self):
        self.last_hid = torch.zeros([self.step + 1, self._n_roll, self._n_agent, self.hidden_size])
        self.count = 0

    ######### 前向传播，做与环境交互的决策时调用
    def forward(self, obs, rnn_states, masks,
                available_actions=None, deterministic=False, hid_state=None):
        ############输入动作：当前每个智能体的局部观测（6*n）+与所有智能体的距离（n）
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        self.count += 1
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        # print(obs.shape)
        if self._use_attn:
            ############# 如果用att，则按照距离排序后的局部观测obs_in和dis_in、hid_state作为输入
            ############# 每个智能体输入的hid_state的维度为(n_agent, hidden_size)，因此总共(n_roll*n_agent, n_agent, hidden_size)
            hid_state = check(hid_state).to(**self.tpdv)
            obs_dis = obs[:, -self._n_agent:]
            obs_in = obs[:, :-self._n_agent]
            dis_in = obs_dis.sort(dim=1)[0]
            ############# 与环境交互的时候不需要进行估计网络的更新，因此不输出估计值
            attn_features, _ = self.base(obs=obs_in, last_hid=hid_state, dis=dis_in)
            resd = self.resnet_mlp(obs_in)
            wei = self.resnet(obs=obs_in)
            actor_features = wei * attn_features.detach() + resd    # n*256
            #actor_features = wei * attn_features + resd
            # actor_features = actor_features.detach()
            # hid_state = actor_features.reshape(self._n_roll, self._n_agent, -1).repeat(1, self._n_agent, 1) \
            #     .reshape(self._n_roll * self._n_agent, self._n_agent, self._att_hidden)
            
            ########################################################
            ############ 输出的actor_features维度为(进程数n_roll * n_agent, hidden_size)
            ############ 需要调整为(n_roll, n_agent, n_agent, hidden_size)来存入buffer，方便进行超出距离限制的hidden的mask操作
            hid_s_ori = actor_features.reshape(self._n_roll, self._n_agent, -1)
            hid_s_t = hid_s_ori.repeat(1, self._n_agent, 1)\
                .reshape(self._n_roll, self._n_agent, self._n_agent, self.hidden_size)
            index_dis = obs_dis.argsort(dim=1).reshape(self._n_roll, self._n_agent, -1)
            att_dis = obs_dis.reshape(self._n_roll, self._n_agent, -1)
            
            ############ 每个进程分开，每个智能体获得的尺寸为(n_agent, hidden_size)的隐藏层根据距离进行排序和重组
            for i in range(self._n_roll):
                hid_temp = deepcopy(hid_s_ori[i])
                for agent in range(self._n_agent):
                    inx_temp = index_dis[i, agent, :]
                    for a in range(self._n_agent):
                        #print(att_dis[i, agent, inx_temp[a]])
                        if att_dis[i, agent, inx_temp[a]] < self.cl or a < 2: #< self._n_agent/2:
                            #hid_s_t[i, agent, inx_temp[a], :] = hid_temp[inx_temp[a]]
                            hid_s_t[i, agent, a, :] = hid_temp[inx_temp[a]]
                        else:
                            #hid_s_t[i, agent, inx_temp[a], :] = torch.zeros(self._att_hidden)
                            hid_s_t[i, agent, a, :] = torch.zeros(self._att_hidden)
            hid_state = hid_s_t.reshape(self._n_roll * self._n_agent, self._n_agent, self.hidden_size)

        # else:
        #     ############# 如果不用att，则局部观测obs_in作为输入
        #obs = obs[:, :]
        #
        actor_features = self.base(obs)
        #
        #     hid_s_ori = actor_features.reshape(self._n_roll, self._n_agent, -1)
        #     hid_s_t = hid_s_ori.repeat(1, self._n_agent, 1) \
        #         .reshape(self._n_roll, self._n_agent, self._n_agent, self.hidden_size)
        #     hid_state = hid_s_t.reshape(self._n_roll * self._n_agent, self._n_agent, self.hidden_size)
        # print(actor_features.shape)
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs#, rnn_states#, hid_state

    ########## 进行训练时的动作评估，用于计算PPO算法中的新老策略区别
    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None, hid_state=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param dis: attn weight
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        if self._use_attn:
            ################################ppo epoch
            hid_state = check(hid_state).to(**self.tpdv)
            obs_dis = obs[:, -self._n_agent:]
            obs_in = obs[:, :-self._n_agent]
            dis_in = obs_dis.sort(dim=1)[0]
            ################### 训练时进行估计网络的梯度更新，因此需要计算全局状态的估计值
            ################### 为了方便就不在函数最后return，而是采用直接从r_mappo的训练模块中读取self.policy.actor.est_temp
            attn_features, self.est_temp = self.base(obs=obs_in, last_hid=hid_state, dis=dis_in)
            resd = self.resnet_mlp(obs_in)
            wei = self.resnet(obs=obs_in)
            actor_features = wei * attn_features.detach() + resd
            #actor_features = wei * attn_features + resd
            #resd, wei = self.resnet(obs=obs_in)
            self.wei = wei.detach().cpu().numpy().mean()
            self.hid_fea = actor_features
            #actor_features = actor_features.detach()
            '''attn_obs = obs.reshape(self._n_roll, self._n_agent, -1)
            for i in range(self._n_roll):
                dis = torch.zeros([self._n_agent, self._n_agent])
                for cx in range(self._n_agent):
                    for cy in range(self._n_agent):
                        dis[cx][cy] = torch.norm(attn_obs[i, cx, 0:3] - attn_obs[i, cy, 0:3])
                self.curr_hid_tr[i], self.est_temp[i] = self.base(obs=attn_obs[i], last_hid=self.last_hid[i], dis=dis)
            self.est_features = self.est_temp.reshape(self._n_roll * self._n_agent, -1)
            actor_features = self.curr_hid_tr.reshape(self._n_roll * self._n_agent, -1)'''
        else:
            #obs = obs[:, :-self._n_agent]
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # 这一次的action evaluate目的是
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

############# Critic架构，比较常规
class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu"),num=0):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        ########## 没有用popart
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
