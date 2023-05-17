#from madrl_environments.pursuit import MAWaterWorld_mod
from asyncio.windows_events import NULL
from MADDPG import MADDPG_SINGLE
import numpy as np
import torch as th
# import visdom
# from params import scale_reward
from make_env import make_env
from copy import deepcopy
import numpy as np
# do not render the scene

e_render = True
food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
n_agents = 5
cl=10
reward_record = []
world = make_env('simple_spread',num=n_agents,cl=cl, benchmark=False, discrete_action=False, discrete_action_input=False)
n_actions = world.action_space[0].shape[0]
n_states = world.observation_space[0].shape[0]

capacity = 50000
batch_size = 512

n_episode = 500
max_steps = 200
episodes_before_train = 30

win = None
param = None
new = True
lr = 1e-5
maddpg = MADDPG_SINGLE(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train, new, lr)
#maddpg.load_model()
FloatTensor = th.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    total_reward = np.zeros(n_agents)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    c_loss_avg = 0.0
    a_loss_avg = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        world.render()
        # if e_render and i_episode > 130 and i_episode % 50 ==0:
        #     world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        input_act = deepcopy(action.numpy())
        #action = maddpg.select_action(obs)
        #obs_, reward, done, _ = world.step(action.numpy())
        #reward = th.FloatTensor(reward).type(FloatTensor)
        obs_, reward_, _, _ = world.step(input_act)
        reward = th.FloatTensor(reward_).type(FloatTensor)
        
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward[0].numpy()
        #rr += reward.cpu().numpy()
        #for i in range(hd_temp.shape[0]):
        #            kd_buffer.push(hd_temp[i], hd_next_temp[i],
        #                           obs_temp[i])
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
        if c_loss != None and a_loss != None:
            c_loss_avg += np.array([item.detach().numpy() for item in c_loss]).sum()
            a_loss_avg += np.array([item.detach().numpy() for item in a_loss]).sum()
        
    maddpg.episode_done += 1
    print('Epi: %d, total = %f' % (i_episode, total_reward))
    print('c_loss_avg = %f, a_loss_avg = %f' % (c_loss_avg, a_loss_avg))
    
    reward_record.append(total_reward)
    if i_episode % 20 == 0 and i_episode>=100:
        maddpg.memory.save_dataset()
        maddpg.savemodel()
    

world.close()
