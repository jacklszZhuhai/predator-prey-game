import re
import numpy as np
import matplotlib.pyplot as plt
pattern = re.compile('is ')
s = 10
ep = 1000
b_n = 0

log_path = [
    "hd003formthres_5v1_vel01506_collide_001"
            ]
name = log_path

inx = 0
lent = int(ep / s)
x = [i*s for i in range(int(b_n / s), lent)] # x轴

for item in log_path:
    path = "./Training_log/0517/" + item + ".log"
    a = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            match = pattern.search(line)
            result = pattern.findall(line)
            if match:
                count += 1
                #print(line)
                if len(line.split('rewards is ')) > 1:
                    tep = line.split('rewards is ')[1].split('\n')[0]
                else:
                    tep = 0
                #print(tep)
                if count > b_n:
                    a.append(float(tep))
                if count == ep:
                    break
        while count < ep:
            a.append(-500.)
            count += 1
    a_n = np.array(a).reshape(-1, s)
    a_m = np.mean(a_n, axis=1)  # 均值
    a_s = np.std(a_n, axis=1)  # 方差

    r1 = list(map(lambda x: x[0] - x[1], zip(a_m, a_s)))
    r2 = list(map(lambda x: x[0] + x[1], zip(a_m, a_s)))
    plt.plot(x, a_m, label=name[inx])
    plt.fill_between(x, r1, r2, alpha=0.2)
    inx += 1

plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Average reward')
plt.show()
plt.savefig('./Training_log/0517/hd003formthres_5v1_vel01506_collide_001')