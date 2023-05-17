from collections import namedtuple
import random
import numpy as np
import os
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))
from copy import deepcopy

class ReplayMemory:
    def __init__(self, capacity, new):
        self.capacity = capacity
        log_dir = "./dataset.npz"
        # 如果有保存的模型，则加载模型，并在其基础上继续训练
        if os.path.exists(log_dir) and new == False:
            load_data = np.load(log_dir,allow_pickle=True)  # 读取含有多个数组的文件
            self.memory = list(load_data['dataset'])
        else:
            self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = deepcopy(Experience(*args))
        #if self.position % 1000:
        #    print(Experience(*args)) 
        self.position = (self.position + 1) % self.capacity
    
    def save_dataset(self):
        full_dataset = np.array(self.memory[-50000:])
        np.savez('./dataset_5', dataset=full_dataset)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
