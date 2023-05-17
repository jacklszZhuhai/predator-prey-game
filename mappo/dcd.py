import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def assi(x,gt):
    if type(x).__name__=='ndarray':
        x = torch.from_numpy(x).float()
        gt = torch.from_numpy(gt).float()
    n_x, _ = x.shape
    dist = torch.zeros([n_x, n_x])
    for i in range(n_x):
        tep = torch.norm(gt - x[i, :].unsqueeze(dim=0), dim=1)
        dist[i, :] = tep.transpose(0,1)

    row_ind, col_ind = linear_sum_assignment(dist)
    cost = dist[row_ind, col_ind].sum()
    return cost


def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    if type(x).__name__=='ndarray':
        x = torch.from_numpy(x)
        gt = torch.from_numpy(gt)
    x = x.unsqueeze(dim=0)
    gt = gt.unsqueeze(dim=0)
    x = x.float()
    gt = gt.float()
    _, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = calc_cd(x, gt)
    h_d = torch.max(dist1.max(dim=1)[0],dist2.max(dim=1)[0])
    cd = (dist1.sum(dim=1) + dist2.sum(dim=1))
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = loss
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    hd_cd = torch.cat([dist1, dist2]).max(dim=0)[0]
    return res.numpy(), cd.numpy(), h_d.numpy(), hd_cd.sort(dim=0)[0].numpy()

def calc_cd(output, gt):
    if type(output).__name__=='ndarray':
        output = torch.from_numpy(output)
        gt = torch.from_numpy(gt)
    output = output.unsqueeze(dim=0)
    gt = gt.unsqueeze(dim=0)
    batchsize, n_x,_ = output.shape
    dist1 = torch.zeros([batchsize, n_x])
    idx1 = torch.zeros([batchsize, n_x])
    dist2 = torch.zeros([batchsize, n_x])
    idx2 = torch.zeros([batchsize, n_x])
    for i in range(n_x):
        tep1 = torch.norm(gt - output[:,i].unsqueeze(dim=1), dim = 2)
        min_temp = torch.min(tep1, dim=1)
        dist1[:,i] = min_temp[0]
        idx1[:,i] = min_temp[1]

        tep2 = torch.norm(output - gt[:,i].unsqueeze(dim=1), dim = 2)
        min_temp = torch.min(tep2, dim=1)
        dist2[:,i] = min_temp[0]
        idx2[:,i] = min_temp[1]
        #print(tep1)
        #print(tep2)

    #print(dist1, idx1)
    #print(dist2, idx2)
    cd = (dist1.sum(dim=1) + dist2.sum(dim=1)).numpy()
    return cd, idx1, idx2


# 打印二维数组的函数，用以显示计算过程
def printTable(Data):
    for i in range(len(Data[0])):
        for y in range(len(Data)):
            if y == len(Data) - 1:
                dot = '\n'
            else:
                dot = ' '
            print(Data[y][i], end=dot)
# 分支算法执行类
class Worker:
    # 初始化
    def __init__(self, matrix):
        self.maxi = 0  # 上界通过贪心算法找出近似值
        self.pt_nodes = []  # 存放可扩展的节点
        self.pt_flag = 0  # 标记队列是否被使用,用于结束算法
        self.min_leaf_node = None  # 消耗最小的叶子节点
        self.n = len(matrix)
        self.matrix = matrix
        self.get_up_limit()
    # 贪心算法获取花费时间的数据上界
    def get_up_limit(self):
        # 初始化n个作业被分配情况
        worker_mark = [0] * self.n
        cost = np.zeros(self.n)
        for i in range(self.n):
            worker_mark.append(0)
        # 利用贪心算法，从第1个作业开始寻找到完成它花费时间最少的工作人员，并记录，最终取得近似最优解
        for i in range(self.n):  # 循环遍历n个作业
            temp = self.matrix[i]  # 获得第i个作业被不同的n个工作人员完成的时间数组temp
            min_cost = float("inf")  # 初始化最小时间为inf
            index = 0  # 初始化第i个作业未被分配
            for k in range(self.n):  # 循环n个工作人员，找到完成作业i时间最小的工作人员
                if worker_mark[k] == 0 and min_cost > temp[k]:  # 如果第k个工人未被分配作业并且第i个作业被第k个人员完成的时间小于最小时间时
                    min_cost = temp[k]  # 将当前完成时间赋给最小时间值
                    index = k  # 记录第i个作业被第k个工人完成获得当前最小时间花费
            worker_mark[index] = 1  # 标记被分配作业的工作人员k
            cost[i] = min_cost
            self.maxi += min_cost  # 累积计算上限值
        #print(np.sort(cost))
    # 队列式（FIFO）分支界限算法,得到并输出最终结果
    def branch_limit(self):
        for i in range(self.n):  # 循环n个工作人员
            time = self.matrix[0][i]  # 得到第i个工作人员完成不同作业的时间
            if time <= self.maxi:  # 如果完成时间小于最大值
                node = Node()  # 则创建节点
                # 初始化节点
                node.deep = 0
                node.cost = time
                node.worker = i
                self.pt_nodes.append(node)  # 将节点放入队列

            while (1):
                if len(self.pt_nodes) == 0:  # 如果队列为空
                    break  # 结束循环
                temp = self.pt_nodes.pop(0)  # 队列非空，则遵守先进先出，弹出一个节点
                present_node = temp  # 将当前节点看作父节点
                total_cost = temp.cost  # 获得遍历至当前节点的总时间花费
                present_deep = temp.deep  # 得到当前节点的深度

                # 初始化工人分配标记数组
                worker_mark = []
                for i in range(self.n):
                    worker_mark.append(0)

                # 检查本节点下的作业分配情况
                worker_mark[temp.worker] = 1  # 将被分配的工人标记为1
                while temp.father is not None:  # 如果当前节点有父节点
                    temp = temp.father  # 则取出当前节点的父节点
                    worker_mark[temp.worker] = 1  # 将父节点的标记置为1

                if present_deep + 1 == self.n:  # 如果遍历深度达到n
                    if self.min_leaf_node is None:  # 遍历至最后一个叶子节点不存在
                        self.min_leaf_node = present_node  # 则取当前节点为叶子节点
                    else:
                        if self.min_leaf_node.cost > present_node.cost:  # 如果叶子节点时间花费大于父节点
                            self.min_leaf_node = present_node  # 则取父节点为叶子节点
                else:
                    # 否则获取当前节点的子节点
                    children = self.matrix[present_deep + 1]
                    # 检查本节点的子节点是否满足进入队列的要求
                    for k in range(self.n):
                        if children[k] + total_cost <= self.maxi and worker_mark[k] == 0:  # 如果时间花费小于最小值，并且第k个人未被分配工作
                            node = Node()  # 则创建节点
                            node.deep = present_deep + 1
                            node.cost = children[k] + total_cost  # 将当前时间加上总时间花费计入新的创建节点中
                            node.worker = k  # 添加节点参数
                            node.father = present_node  # 添加节点父节点
                            self.pt_nodes.append(node)  # 将节点放入队列

        # 输出算法执行的结果
        temp = self.min_leaf_node
        return temp

# 分支节点类
class Node:
    def __init__(self):
        self.deep = 0  # 标记该节点的深度
        self.cost = 0  # 标记到达该节点的总时间花费
        self.father = None  # 标记该节点的父节点
        self.worker = None  # 本节点的该任务由第几位工人完成


def cacl_worker(x, gt):
    n = x.shape[0]
    matrix = np.zeros([n, n])
    maxi = 0
    t_d = np.zeros(n)
    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.linalg.norm(gt[i] - x[j])
    worker_mark = [0] * n
    #for i in range(n):
    #    worker_mark.append(0)
    # 利用贪心算法，从第1个作业开始寻找到完成它花费时间最少的工作人员，并记录，最终取得近似最优解
    for i in range(n):  # 循环遍历n个作业
        temp = matrix[i]  # 获得第i个作业被不同的n个工作人员完成的时间数组temp
        min_cost = float("inf")  # 初始化最小时间为inf
        index = 0  # 初始化第i个作业未被分配
        for k in range(n):  # 循环n个工作人员，找到完成作业i时间最小的工作人员
            if worker_mark[k] == 0 and min_cost > temp[k]:  # 如果第k个工人未被分配作业并且第i个作业被第k个人员完成的时间小于最小时间时
                min_cost = temp[k]  # 将当前完成时间赋给最小时间值
                index = k  # 记录第i个作业被第k个工人完成获得当前最小时间花费
        worker_mark[index] = 1  # 标记被分配作业的工作人员k
        t_d[i] = min_cost
        # maxi += min_cost
    print(t_d.max())
    worker = Worker(matrix=matrix)
    #  执行分支界限算法
    temp = worker.branch_limit()
    for i in range(n):
        # print('第' + str(temp.worker + 1) + '位工人被分配了第' + str(temp.deep + 1) + '份工作')
        # print(temp.cost)
        t_d[i] = matrix[temp.deep][temp.worker]
        temp = temp.father  # 返回父节点
    return np.sort(t_d)


if __name__ == "__main__":
    import time

    t0 = time.time()
    a = torch.randint(-4,4,[10,2])
    #b = torch.randint(0,10,[1,9,3])

    b = a + torch.normal(0,0.2,(10,2))
    #b = a
    #b[1:5] = a[1:5]
    #b[0] = (a[0]+2*a[1])/3
    #c = b[:,torch.randperm(5),:]
    dcd1,cd1,hd_1,_ = calc_dcd(a,b,alpha=0.8, n_lambda=2)
    print(_)
    #print(b)

    #dcd2, cd2,_,_ = calc_dcd(a, b, alpha=1, n_lambda=2)
    #dcd2 = calc_dcd(a,c,alpha=1)
    #dcd_base = calc_dcd(a,a,alpha=1)
    #print(a)
    t1 = time.time()
    print(t1-t0)
    #print(c)
    #print(dcd1,cd1)
    #print(dcd2,cd2)
    #print(dcd_base)
    if hd_1[0]<1:
        num = 10
        #a = torch.randint(-4, 4, [num,2])
        #b = torch.randint(-4,4,[10,2])#a + torch.normal(0, 0.1, (num,2))

        #b[0]=a[1]
        x = np.array(a)
        gt = np.array(b)#torch.randint(-4, 4, [num, 2]))
        matrix = np.zeros([num, num])
        t_d = np.zeros(num)
        for i in range(num):
            for j in range(num):
                matrix[i][j] = np.linalg.norm(x[i]-gt[j])
        #matrix = np.array(torch.randint(0, 10, [num, num]))
        #print(matrix)
        #  初始化算法执行类
        worker = Worker(matrix)
        #  执行分支界限算法
        temp = worker.branch_limit()
        #dcd1, cd1, hd_1, _ = calc_dcd(a, b, alpha=1, n_lambda=2)
        #print(cd1,hd_1, _)
        #print(temp.cost)
        #print('第' + str(temp.worker + 1) + '位工人被分配了第' + str(temp.deep + 1) + '份工作')
        i=0
        t_d[i] = matrix[temp.deep][temp.worker]
        while temp.father is not None:  # 如果存在父节点
            temp = temp.father  # 返回父节点
            #print('第' + str(temp.worker + 1) + '位工人被分配了第' + str(temp.deep + 1) + '份工作')
            #print(temp.cost)
            i += 1
            t_d[i] = matrix[temp.deep][temp.worker]
        print(t_d)
    #print(np.sort(t_d)[-2:])
    t2 = time.time()
    print(t2-t1)