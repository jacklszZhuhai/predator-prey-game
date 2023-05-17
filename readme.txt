train.py 是训练的主要函数，本来应该在mappo的train文件里的，但为了方便跑我就把它拎到mappo同级文件来了
multiagent-particle-envs文件夹是之前发的MPE环境，我在里头的multiagent/scenarios/simple_spread.py添加了注释，设置成了无序编队的，并且为了适应mappo的输入也对simple_spread的obs输出格式做了修改
mappo文件夹是核心模块：
1. sp_env.py：MPE环境预处理文件
2. config.py：参数配置文件
3. algorithms文件夹：主要mappo算法部分，很重要
4. envs文件夹：只用到了env_wrappers.py，但不看也没事，没有修改和注释
5. runner文件夹：其中的shared文件夹内两个都是最主要的运行文件，其中base_runner是env_runner的父类，主要的运行代码在env_runner里面，很重要，加载或保存模型在baserunner里
6. scripes文件夹：不用看
7. train文件夹：一些我之前用来测试环境代码和搞过策略蒸馏的代码，不看没事，还没加注释
8. utils文件夹：里面只用了shared_buffer.py和util.py，其他mappo中没用，其中shared_buffer.py是经验池和计算GAE的需要看看，util.py比较杂，粗看下也没事，没加注释