"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""

import torch
from mappo.algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from mappo.utils.util import update_linear_schedule


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), num=None):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.Num_agent = num
        self.obs_space = obs_space
        self.share_obs_space = obs_space #cent_obs_space
        self.act_space = act_space
        print(self.device)
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device, self.Num_agent)
        self.critic = R_Critic(args, self.share_obs_space, self.device, self.Num_agent)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self._use_attn = args.use_attn

    '''
        if self._use_attn:
            from mappo.algorithms.utils.att import Estimate
            self.est = Estimate(d_model=128, obs_space=self.share_obs_space)
            self.est_optimizer = torch.optim.Adam(self.est.parameters(),
                                                  lr=self.lr, eps=self.opti_eps,
                                                  weight_decay=self.weight_decay)

    def train_est(self, est_input, share_obs):
        # input_hid = est_input.detach().copy()
        est_obs = self.est(est_input)
        gt = torch.tensor(share_obs)
        est_loss = torch.nn.MSELoss()(gt, est_obs)
        self.est_optimizer.zero_grad()
        est_loss.requires_grad_(True)
        est_loss.backward()
        self.est_optimizer.step()
    '''

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        '''if self._use_attn:
            update_linear_schedule(self.est_optimizer, episode, episodes, self.critic_lr)'''
    ################## 与环境交互时返回动作
    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks, hid_state=None, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # print(self.actor)
        # print(self.critic)
        actions, action_log_probs = self.actor(obs,
                                             rnn_states_actor,
                                             masks,
                                             available_actions,
                                             deterministic,
                                             hid_state)

        '''if self._use_attn:
            self.train_est(est_input=self.actor.est_input, share_obs=cent_obs)'''
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    ################ 返回值函数
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    ################ 训练时评估动作和生成全局信息估计
    def evaluate_actions(self,obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, hid_state=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                                                     hid_state)

        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    ######### 没用上
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


