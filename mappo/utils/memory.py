from collections import namedtuple
import random
import numpy as np
import os

Experience = namedtuple('Experience',
                        ('hidden', 'hidden_next', 'obs_total'))


class ReplayMemory:
    def __init__(self, capacity, save_dir, new):
        self.capacity = capacity
        self.log_dir = str(save_dir) + "/dataset.npz"
        # 如果有保存的模型，则加载模型，并在其基础上继续训练
        if os.path.exists(self.log_dir) and new == False:
            load_data = np.load(self.log_dir, allow_pickle=True)  # 读取含有多个数组的文件
            self.memory = list(load_data['dataset'])
            print('load dataset')
        else:
            self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        # if self.position % 1000:
        #    print(Experience(*args))
        self.position = (self.position + 1) % self.capacity

    def save_dataset(self):
        full_dataset = np.array(self.memory)
        np.savez(self.log_dir, dataset=full_dataset)

    def sample(self, batch_size):
        temp = random.sample(self.memory, batch_size)
        mini_batch = Experience(*zip(*temp))
        #state_batch = th.stack(batch.states).type(FloatTensor).squeeze(0)
        #action_batch = th.stack(batch.actions).type(FloatTensor).squeeze(0)
        #reward_batch = th.stack(batch.rewards).type(FloatTensor).squeeze(0)
        return mini_batch

    def __len__(self):
        return len(self.memory)

    def comm_limit(self, comm_dis, num):
        self.me_temp = []
        for i, item in enumerate(self.memory):
            obs_temp = np.array(item[2])
            obs_in = obs_temp[:-num]
            obs_dis = obs_temp[-num:]
            index_dis = obs_dis.argsort()
            obs_in = obs_in.reshape(num, -1)
            obs_copy = np.zeros_like(obs_in)
            hid_temp = np.array(item[0])
            hid_copy = np.zeros_like(hid_temp)

            for a in range(num):
                if obs_dis[index_dis[a]] < comm_dis or a < 2:
                    hid_copy[index_dis[a]] = hid_temp[index_dis[a]]
                    obs_copy[index_dis[a]] = obs_in[index_dis[a]]
                else:
                    hid_copy[index_dis[a]] = -1e9
                    obs_copy[index_dis[a]] = -30
            obs_copy = np.concatenate((obs_copy.flatten(),obs_dis))
            obs_copy = np.vstack((obs_copy,obs_temp))
            self.memory[i][0] = hid_copy
            self.memory[i][1] = item[1]
            self.memory[i][2] = obs_copy
            # self.memory[i] = Experience(hid_copy, item[1], obs_copy)
