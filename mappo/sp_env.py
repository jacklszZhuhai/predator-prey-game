import gym
import numpy as np
from copy import deepcopy
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import spaces

### 环境配置文件
# 创建simple spread环境，需要去simple spread安装的文件那边修改simple_spread.py才能自定义环境
def m_env(args, scenario_name, num_prey=4, num_predator=1, benchmark=False,
          discrete_action=False, discrete_action_input=False
               ):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(arg=args, num_prey=num_prey, num_predator = num_predator)
    if benchmark:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            info_callback=scenario.benchmark_data,
                            action_callback=scenario.action
                            )
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            )
    return env

# 环境包装
class EnvCore(object):

    def __init__(self, args):
        self.agent_num = args.num_prey
        self.gui = True
        self.world =  m_env(args=args, scenario_name='simple_tag',num_prey=args.num_prey,num_predator=args.num_predator,
                             benchmark=False, discrete_action=False,
                             discrete_action_input=False)
        self.action_dim = self.world.action_space[0].shape[0]
        #self.obs_dim = self.world.observation_space[0].shape[0] - self.agent_num
        self.obs_dim = 4 * args.num_prey + 2 * args.num_predator
        self.single_dim = 4
        self.cl = args.comm_dis
    
    # 对simple spread输出的状态进行处理，虽然share_state没用，只是为了方便在mappo中对齐维度省得改之前的代码
    def obs_re(self, obs):
        share_state = np.zeros([self.agent_num, self.single_dim])
        # De = np.zeros(self.agent_num, 1).squeeze()
        for i in range(self.agent_num):
            share_state[i, :] = obs[i][:4]

        share_state = share_state.reshape(-1)
        
        return obs, share_state
    
    def reset(self):
        obs = self.world.reset()
        #real_state, share_obs = self.obs_re(obs)
        #sub_agent_obs = []
        #share_obs = np.concatenate([share_obs, np.zeros(self.world.observation_space[0].shape[0] - self.obs_dim)])
        share_obs = obs
        #sub_agent_obs.append(share_obs)
        # for i in range(self.agent_num):
        #     sub_obs = real_state[i]  # OBS[str(i)]["state"]
        #     sub_agent_obs.append(sub_obs)
        return share_obs
    
    def step(self, actions):
        input_act = deepcopy(actions)
        obs_, reward_, _, _ , rew_unit_buff_= self.world.step(input_act)
        #self.world.render()
        #real_state, share_obs = self.obs_re(obs_)
        real_state = obs_
        share_obs = obs_            # 认为所观测即为当前状态不作区分
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        #share_obs = np.concatenate([share_obs, np.zeros(self.world.observation_space[0].shape[0] - self.obs_dim)])
        #sub_agent_obs.append(share_obs)
        sub_agent_rew_unit_buff = []
        for i in range(self.agent_num):
            sub_agent_obs.append(real_state[i])
            sub_agent_reward.append([reward_[i]])
            sub_agent_done.append(False)
            sub_agent_info.append({})
            sub_agent_rew_unit_buff.append(rew_unit_buff_[i])
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_agent_rew_unit_buff]

    def close(self):
        self.world.close()

# 这部分不看也问题不大
class PybulletEnv(object):
    """对于离散动作环境的封装"""
    def __init__(self, args):
        self.env = EnvCore(args)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        #self.action_space.append(3)
        share_obs_dim = 0
        #print(0)
        for agent in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_input:
                # physical action space
                u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作
            else:
                u_action_space = spaces.Box(low=-1., high=1., shape=(self.signal_action_dim,),
                                                     dtype=np.float32)

            if self.movable:
                total_action_space.append(u_action_space)
            #print(len(total_action_space))
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    #print(1)
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    #print(2)
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            # share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-10., high=10., shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]
        share_obs_dim = self.signal_obs_dim
        self.share_observation_space = [spaces.Box(low=-10., high=10., shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]
        #print(self.observation_space)
        #print(self.action_space)
        #print(self.observation_space)
    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos, rews_unit = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(rews_unit)

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

#这下面的没用到
class MultiDiscrete(gym.Space):


    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

if __name__ == "__main__":
    PybulletEnv().step(actions=None)