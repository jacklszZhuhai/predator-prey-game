from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import os
scale_reward = 0.01
def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG_SINGLE:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train, new=True, learn_rate = 0.001):
        #self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        #self.critics = [Critic(n_agents, dim_obs,
        #                       dim_act) for i in range(n_agents)]
        self.actors = Actor(dim_obs, dim_act)
        self.critics = Critic(n_agents, dim_obs, dim_act)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.new = new
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act 
        self.memory = ReplayMemory(capacity, new)
        self.batch_size = batch_size
        self.use_cuda = False#th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        self.lr = learn_rate
        self.GAMMA = 0.95
        self.tau = 0.01
        self.save_dir = './'
        self.var = 1.0
        self.critic_optimizer = Adam(self.critics.parameters(), lr=self.lr)
        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.lr)
        #self.critic_optimizer = [Adam(x.parameters(),
        #                              lr=self.lr) for x in self.critics]
        #self.actor_optimizer = [Adam(x.parameters(),
        #                             lr=self.lr) for x in self.actors]

        if self.use_cuda:
            self.actors = self.actors.cuda()
            self.critics = self.critics.cuda()
            self.actors_target = self.actors_target.cuda()
            self.critics_target = self.critics_target.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train and self.new:
            return None, None
        ByteTensor = th.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))
        non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                batch.next_states)))
        # state_batch: batch_size x n_agents x dim_obs
        state_batch = th.stack(batch.states).type(FloatTensor)
        action_batch = th.stack(batch.actions).type(FloatTensor)
        reward_batch = th.stack(batch.rewards).type(FloatTensor)
        # : (batch_size_non_final) x n_agents x dim_obs
        non_final_next_states = th.stack(
            [s for s in batch.next_states
                if s is not None]).type(FloatTensor)
        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)
        self.critic_optimizer.zero_grad()
        current_Q = self.critics(whole_state, whole_action)
        
        non_final_next_actions = [
            self.actors_target(non_final_next_states[:,i,:]) for i in range(
                                                            self.n_agents)]
        
        non_final_next_actions = th.stack(non_final_next_actions)
        #non_final_next_actions = (
        #    non_final_next_actions.transpose(0,
        #                                     1).contiguous())

        target_Q = th.zeros(
            self.batch_size).type(FloatTensor)
        #print(non_final_next_actions.view(-1, self.n_agents * self.n_actions).size())
        target_Q[non_final_mask] = self.critics_target(
            non_final_next_states.view(-1, self.n_agents * self.n_states),
            non_final_next_actions.view(-1, self.n_agents * self.n_actions)
            #non_final_next_actions.view(-1)
        ).squeeze()
        # scale_reward: to scale reward in Q functions

        target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
            reward_batch[:, 0].unsqueeze(1) * scale_reward)

        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        loss_Q.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        loss_abatch = th.zeros(self.n_agents)
        for agent in range(self.n_agents):
            state_i = state_batch[:, agent, :]
            action_i = self.actors(state_i)
            #print(action_i)
            ac = action_batch.clone()
            ac[:, agent] = action_i.view(self.batch_size, -1)
            #whole_action = ac
            whole_action = ac.view(self.batch_size, -1)
            #whole_action = ac.view(-1, self.n_agents * self.n_actions)
            actor_loss = -self.critics(whole_state, whole_action)
            loss_abatch[agent] = actor_loss.mean()
        toa_loss = loss_abatch.mean()
        toa_loss.backward()
        self.actor_optimizer.step()
        c_loss.append(loss_Q)
        a_loss.append(toa_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            soft_update(self.critics_target, self.critics, self.tau)
            soft_update(self.actors_target, self.actors, self.tau)
        #print(c_loss, a_loss)

        return c_loss, a_loss
    
    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(self.n_agents, self.n_actions)
        FloatTensor = th.FloatTensor if self.use_cuda else th.FloatTensor
            
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors(sb.unsqueeze(0)).squeeze()
            act += th.from_numpy(np.random.randn(self.n_actions) * self.var).type(FloatTensor)
            #act[0] = 0
            if self.episode_done > self.episodes_before_train and\
               self.var > 0.05:
                self.var *= 0.98
            act = th.clamp(act, -1, 1)
            actions[i, :] = act
        self.steps_done += 1

        return actions


    def savemodel(self):
        th.save(self.actors.state_dict(), str(self.save_dir) + "actor_9.pt")
        th.save(self.critics.state_dict(), str(self.save_dir) + "critic_9.pt")
    
        
    def load_model(self):
        policy_actor_state_dict = th.load(str(self.save_dir) + 'actor_9.pt', map_location=th.device('cpu'))
        self.actors.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = th.load(str(self.save_dir) + 'critic_9.pt', map_location=th.device('cpu'))
        self.critics.load_state_dict(policy_critic_state_dict)
            
        print('预训练模型加载成功！')


    
       
    