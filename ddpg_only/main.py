import time
from ddpg_only.Model.ddpg_model import DDPG
from Env.env import RLEnv
import numpy as np
import utils.np_reformulate_func as nrf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    '''环境体初始化以及其参数'''
    fap_cnt = 5
    cluster_size = 3
    content_cnt = 1000
    fap_capacity = 20
    is_non_iid = True
    skw_base = 1.2
    skw_sd = 0.5
    skw_scale = 0.1
    plateau = 0.1
    delay_threshold = (0.8, 8.65, 10)
    srv_avg_delay = (0.5, 0.3, 8)
    reward_para = 0.3

    # 训练结构的 hyper parameters
    VAR = fap_capacity * 0.55  # control exploration
    MAX_EPISODES = 250
    MAX_EP_STEPS = 1000
    MEMORY_CAPACITY = 20000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies
    CONF = [0, VAR, MAX_EPISODES, MAX_EP_STEPS, MEMORY_CAPACITY, 0]


    # train



    reward_records = np.zeros(0, dtype=float)

    rl_env = RLEnv(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale,
                   plateau,
                   delay_threshold, srv_avg_delay, reward_para)

    rl_env.init_frans_cache()
    s_dim = rl_env.get_cur_frans_state().shape[0]
    a_dim = 1
    a_bound = rl_env.fap_capacity * 0.55
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=[a_bound],
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = rl_env.reset(CONF)
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            # if RENDER:
            #     env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -a_bound, a_bound) # 在动作选择上添加随机噪声
            real_a = a[0] + a_bound
            s_, r = rl_env.step_by_frans(real_a)
            # print('Episode:', i, " Step:", j, "reward: %f" % r)
            # print("_______________\n")
            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %.2f' % VAR, )
                reward_records = np.append(reward_records, ep_reward)
            CONF[]

    print('Running time: ', time.time() - t1)
    x1, y1 = nrf.partition_avg(reward_records, 1)
    x2, y2 = nrf.partition_avg(rl_env.time_slot_delays, 200)
    x3, y3 = nrf.partition_avg(rl_env.cache_hits, 200)
    x4, y4 = nrf.partition_avg(rl_env.time_out_hits, 200)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1)
    plt.subplot(2, 2, 2)
    plt.plot(x2, y2)
    plt.subplot(2, 2, 3)
    plt.plot(x3, y3)
    plt.subplot(2, 2, 4)
    plt.plot(x4, y4)
    plt.show()
