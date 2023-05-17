"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""
import os

"""
# @Time    : 2021/7/1 7:04 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : huaru_runner.py
"""

import time
import numpy as np
import torch
from mappo.runner.shared.base_runner import Runner
import imageio


def _t2n(x):
    return x.detach().cpu().numpy() if type(x) != np.ndarray else x


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
    ############### 主要环境交互与更新模块
    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        ############### 获得上一时刻信息，生成动作
        # self.model_dir = './Training_Parameter_Storage/0517/cd001formthres_5v1_vel0208_collide'
        # self.policy_preys.actor.load_state_dict(torch.load(str(self.model_dir) + '/actor_prey.pt', map_location=torch.device('cpu')))
        # self.policy_preys.critic.load_state_dict(torch.load(str(self.model_dir) + '/critic_prey.pt', map_location=torch.device('cpu')))

        act_interval = int(self.all_args.act_interval)

        for episode in range(episodes):
            ############ 每次初始化
            self.warmup()
            if self._use_attn:
                self.trainer.policy.actor.init_att()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            # 从已经训练好的模型加载policy参数
            #skip_time = 0
            running_time = 0
            for step in range(self.episode_length):
                if step % act_interval == 0:
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                        actions_env = self.collect(step)

                    # Obser reward and next obs
                    ############### 用动作更新环境
                    obs, rewards, dones, infos , rews_unit= self.envs.step(actions_env)
                    # pos_prey = obs[:, 0, 2:4]
                    # pos_predator = obs[:, self.all_args.num_prey:, 2:4]
                    # predator_out_range = [True if np.sqrt(np.sum(np.square(pos_predator[:,i]))) >= 3 else False for i in range(self.all_args.num_predator)]
                    # if np.sqrt(np.sum(np.square(pos_prey[:,:]))) >= 3 or (True in predator_out_range):
                    #     #skip_time += 1
                    #     break
                    data = obs, rewards, rews_unit, dones, infos, values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic
                # insert data into buffer
                ################ 存入信息
                    self.insert(data)
                    running_time += 1
            # compute return and update network
            ############## 进行参数更新和训练
            # 如果捕食者或被捕食者跑出边界范围，则break这一episode，直接拿前running_time个transitions进行训练
            if running_time == self.episode_length - 1:
                skip_time = -1
            else:
                skip_time = self.episode_length - running_time - 1
            # 如果running_time太小，说明过于鲁莽，给与惩罚。直接在GAE计算return时加上
            reward_in_range = - 0.1 * (self.episode_length - running_time)

            self.compute(skip_time, reward_in_range)
            #train_infos_prey, train_infos_predator = self.train(skip_time)
            train_infos_prey = self.train(skip_time)
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                print("This Episode have store :{} valid steps to train\n".format(running_time))

                train_infos_prey["prey_average_episode_rewards"] = np.mean(self.buffer.rewards[:skip_time]) * self.episode_length
                print("prey average episode rewards is {}".format(train_infos_prey["prey_average_episode_rewards"]))
                print('value: ', train_infos_prey['value_loss'], ' policy:', train_infos_prey['policy_loss'])
                self.log_train(train_infos_prey, total_num_steps)

                train_infos_prey["prey_average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("Each thread episode rewards:")
                for i in range(int(self.all_args.n_rollout_threads)):
                    print(f"thread: {i} total rewards:{np.mean(self.buffer.rewards[:, i, :]) * self.episode_length}")
                    #[rew_forma_base, rew_dis_2_center, distance_reward,  -collide_num * 30, -region_score]:{length, thread, num, 5 item}
                    print(f"   formation shape punish: {np.mean(self.buffer.rewards_unit[:, i, :, 0])* self.episode_length}\
                        \n   scatter punish: {np.mean(self.buffer.rewards_unit[:, i, :, 1])* self.episode_length}\
                        \n   keep away from predator punish: {np.mean(self.buffer.rewards_unit[:, i, :, 2])* self.episode_length}\
                        \n   collide punish: {np.mean(self.buffer.rewards_unit[:, i, :, 3])* self.episode_length}\
                        \n   collide between agent punish:{(np.mean(self.buffer.rewards[:, i, :]) - np.mean(self.buffer.rewards_unit[:, i, :, 0]) - np.mean(self.buffer.rewards_unit[:, i, :, 1]) - np.mean(self.buffer.rewards_unit[:, i, :, 2]) - np.mean(self.buffer.rewards_unit[:, i, :, 3]) - np.mean(self.buffer.rewards_unit[:, i, :, 4])) * self.episode_length}\
                        \n   outside score punish: {np.mean(self.buffer.rewards_unit[:, i, :, 4])* self.episode_length}")
                print(f"Current variance of distribution:{self.policy_preys.actor.act.action_out.epi}")
                print(self.episode_length)
                self.log_train(train_infos_prey, total_num_steps)


                self.buffer.after_update(self.all_args, self.num_agents, self.envs.observation_space[0],
                                         self.envs.observation_space[0], self.envs.action_space[0])
                date = "0517/"
                root_folder = os.getcwd() + '/Training_Parameter_Storage/' + date + self.experiment_name
                if (episode % 100 == 0 or episode == episodes - 1):
                    sub_folder = root_folder #+ '/' + str(episode)
                    if not os.path.exists(sub_folder):
                        os.makedirs(sub_folder)
                    torch.save(self.trainer_prey.policy.actor.state_dict(), sub_folder + "/actor_prey.pt")
                    torch.save(self.trainer_prey.policy.critic.state_dict(), sub_folder + "/critic_prey.pt")

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    ########### 初始化环境
    def warmup(self):
        # reset env
        obs_get = self.envs.reset()  # nr, num+1, 9*num
        self.buffer.preys_obs[0] = obs_get.copy()

    ############## 与环境交互时生成动作
    @torch.no_grad()
    def collect(self, step):
        self.trainer_prey.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer_prey.policy.get_actions(np.concatenate(self.buffer.preys_obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.prey_masks[step]),
                                              np.concatenate(self.buffer.prey_hid_states[step]),)

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # rearrange action
        # print(actions)
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            actions_env = actions
            # raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env#, hid_state

    #########将信息加入buffer
    def insert(self, data):
        obs_get, rewards, rews_unit, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data
        prey_obs = obs_get
        # replay buffer
        # obs = obs_get[:, 1:, :]
        # rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #                                      dtype=np.float32)
        # rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
        #                                             dtype=np.float32)
        ################
        #hid_state[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.att_hidden),
        #                                     dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.all_args.num_prey, 1), dtype=np.float32)
        #masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if self.use_centralized_V:
        #     share_obs = obs_get
        #     #share_obs = obs_get[:, 0, :-self.num_agents]
        #     #share_obs = obs.reshape(self.n_rollout_threads, -1)
        #     #share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # else:
        #     share_obs = obs[:, 1:, :-self.num_agents]

        self.buffer.insert(prey_obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, rewards_unit=rews_unit)

    ##########下面的三个大的函数都是在策略蒸馏的时候用的，现在没什么用
    @torch.no_grad()
    def eval(self):
        for eval_step in range(10):
            self.warmup()
            eval_episode_rewards = []
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                actions_env, hid_state = self.collect(step)

                # Obser reward and next obs
                # print(actions_env)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                eval_episode_rewards.append(rewards)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, hid_state

                # insert data into buffer
                self.insert(data)
            #self.buffer.after_update(self.all_args, self.num_agents, self.envs.observation_space[0], self.envs.observation_space[0], self.envs.action_space[0])
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_env_infos = {}
            eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
            eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
            print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))

    @torch.no_grad()
    def eval_kd(self):
        from mappo.utils.memory import ReplayMemory
        kd_buffer = ReplayMemory(capacity=100000, save_dir=self.save_dir, new=True)

        for eval_step in range(80):
            self.warmup()
            eval_episode_rewards = []
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                actions_env, hid_state = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                eval_episode_rewards.append(rewards)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, hid_state
                self.insert(data)
                #episode_length, n_rollout_threads, num_agents = rewards.shape[0:3]
                #batch_size = n_rollout_threads * episode_length * num_agents

                hd_temp = self.buffer.hid_states[step].reshape(-1, *self.buffer.hid_states[step].shape[2:])
                hd_next_temp = self.buffer.hid_states[step+1, :, 0, :]
                hd_next_temp = hd_next_temp.reshape(-1, *hd_next_temp.shape[2:])
                #hd_next_temp = self.buffer.hid_states[step+1].reshape(-1, *self.buffer.hid_states[step+1].shape[2:])
                obs_temp = self.buffer.obs[step].reshape(-1, *self.buffer.obs[step+1].shape[2:])
                for i in range(hd_temp.shape[0]):
                    kd_buffer.push(hd_temp[i], hd_next_temp[i],
                                   obs_temp[i])

            self.buffer.after_update()
            eval_episode_rewards = np.array(eval_episode_rewards)
            eval_env_infos = {}
            eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
            eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
            print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        kd_buffer.save_dataset()

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=False)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
