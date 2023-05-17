"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch
from mappo.config import get_config
from mappo.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from mappo.sp_env import PybulletEnv

######### 打包（伪）多线程环境，不看也行
def pybullet_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = PybulletEnv(all_args)
            # env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from .mappo.envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

######### 没用上，配置参数全在mappo的config文件里
def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='formation', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)

    all_args = parser.parse_known_args(args)[0]

    return all_args

######### 主函数，但也不需要怎么弄懂
##### run_dir决定参数保存路径，目前的设置有点小问题但因为问题不大就还没改，就是run_dir会在train.py所在文件夹的上一级创立result文件夹来保存
def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:1")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/formation_mappo/results") / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run1'
    #curr_run = 'run_good'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = pybullet_train_env(all_args)
    eval_envs = None#make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    print(num_agents)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from mappo.runner.shared.env_runner import EnvRunner as Runner
    else:
        from mappo.runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    #runner.restore(pre=pre_dir)
    #runner.eval()
    runner.run()

    # post process
    envs.close()
    #if all_args.use_eval and eval_envs is not envs:
    #    eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
