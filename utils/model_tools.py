from ddpg_only.Model.ddpg_model import DDPG
import numpy as np


# 生成一个action
def action_gen(tp: 0, s, a_bound, VAR, cur_ep: int, memory_cap: int, max_ep_steps: int, ddpg: DDPG):
    if tp == 0 :
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, VAR), -a_bound, a_bound)  # 在动作选择上添加随机噪声
    else:
        if cur_ep >= memory_cap / max_ep_steps:
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -a_bound, a_bound)  # 在动作选择上添加随机噪声
        else:  # 基于问题特征，在模型训练初期直接使用平均随机
            a = np.random.rand() * a_bound * 2 - a_bound
    return a[0] + a_bound, a
