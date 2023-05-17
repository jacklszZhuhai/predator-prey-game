import re
import numpy as np
import matplotlib.pyplot as plt
pattern = re.compile('is ')
s = 2   # 10
ep = 64 # 640
#log_path = ["attn_a5_n4_ep15_lre5_cl3_2_est005",
#            "attn_a5_n4_ep15_lre5_cl3_4_est020",
#            "noattn_a5_n4_ep15_lre5_cl3_2",
#            ]
#name = ["5 agent attn est020",
#        "5 agent attn est005",
#        "5 agent no attn"
#        ]
#log_path = ["attn_a9_n4_ep15_lre5_cl3_2_est020",
#            "attn_a9_n8_ep15_lre6_cl3_2",
#            "attn_a9_n4_ep15_lre6_cl3_2_est0",
#            "noattn_a9_n4_ep15_lre6_cl3_2",
#            ]
#name = ["9 agent attn est020",
#        "9 agent attn est005",
#        "9 agent attn est000",
#        "9 agent no attn"
#        ]
#log_path = ["attn_a9_n4_ep15_lre5_cl3_5_est050",
#            "attn_a9_n4_ep15_lre5_cl3_5_est020",
#            "attn_a9_n4_ep15_lre5_cl3_5_est010",
#            "attn_a9_n4_ep15_lre5_cl3_5_est000",
#            "noattn_a9_n4_ep15_lre5_cl3_5",
#            ]
log_path = [
    "no_a8_256_base_cl0_1",


            ]
name = [
    "no_a8_256_base_cl0",

        ]
inx = 0
lent = int(ep / s)
x = [i*s for i in range(lent)] # x轴

for item in log_path:
    path = "./0411/" + item + ".txt"
    a = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            match = pattern.search(line)
            result = pattern.findall(line)
            if match:
                count += 1
                tep = line.split('is ')[1].split('\n')[0]
                a.append(float(tep))
                if count == ep:
                    break
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