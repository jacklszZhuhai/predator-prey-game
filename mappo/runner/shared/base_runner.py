import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from mappo.utils.shared_buffer import SharedReplayBuffer
from mappo.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
from mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self._use_attn = self.all_args.use_attn
        self.att_hidden = self.all_args.att_hidden
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy_preys = Policy(self.all_args,
                                   self.envs.observation_space[0],
                                   self.envs.observation_space[0],
                                   self.envs.action_space[0],
                                   device = self.device
                                   )

        if self.model_dir is not None:
            self.restore()
        ############################# 利用r_mappo文件进行训练
        # algorithm
        self.trainer_prey = TrainAlgo(self.all_args, self.policy_preys, device = self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.all_args.num_prey,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self, skip_time, reward_out_range):
        """Calculate returns for the collected data."""
        self.trainer_prey.prep_rollout()
        if skip_time != -1:
            last_index = -1 * skip_time
        else:
            last_index = skip_time
        # prey根据最后一刻状态求GAE
        next_values = self.trainer_prey.policy.get_values(np.concatenate(self.buffer.preys_obs[last_index]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.prey_masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        # self.buffer.compute_returns(next_values, self.trainer_prey.value_normalizer, flag=0, skip_time=last_index)
        # # predator根据最后一刻状态求GAE
        # next_values = self.trainer_predator.policy.get_values(np.concatenate(self.buffer.predators_obs[last_index]),
        #                                                   np.concatenate(self.buffer.rnn_states_critic[-1]),
        #                                                   np.concatenate(self.buffer.predators_masks[-1]))
        # next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer_prey.value_normalizer, skip_time=last_index, reward_out_range=reward_out_range)

    def train(self, skip_time=-1):
        """Train policies with data in buffer. """
        if skip_time!= -1:
            last_index = -1 * skip_time
        else:
            last_index = skip_time
        self.trainer_prey.prep_training()
        train_infos_prey = self.trainer_prey.train(self.buffer,  last_index = last_index)
        # self.trainer_predator.prep_training()
        # train_infos_predator = self.trainer_predator.train(self.buffer, flag=1, last_index = last_index)
        #self.buffer.after_update(self.all_args, self.num_agents, self.envs.observation_space[0], self.envs.observation_space[0], self.envs.action_space[0])

        #return train_infos_prey, train_infos_predator
        return train_infos_prey
    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        print(str(self.save_dir))
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, pre=None):
        """Restore policy's networks from a saved model."""
        if pre is None:
            dir = self.save_dir
        else:
            dir = pre
        self.model_dir = dir
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=torch.device('cpu'))
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=torch.device('cpu'))
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
