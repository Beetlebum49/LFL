import Env.env as env
import Env.fap as fap
from matplotlib import pyplot as plt
import numpy as np
import utils.np_reformulate_func as nrf

fap_cnt = 5
cluster_size = 3
content_cnt = 400
fap_capacity = 15
is_non_iid = True
skw_base = 1.1
skw_sd = 1
skw_scale = 0.3
plateau = 0.1
delay_threshold = (0.8, 8.65, 10)
srv_avg_delay = (0.5, 0.3, 8)
reward_para = 0.3

rl_env = env.RLEnv(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale, plateau,
              delay_threshold, srv_avg_delay,reward_para)

rl_env.init_frans_cache()
for i in range(1, rl_env.fap_cnt+1):
    f: fap.FAP
    f = rl_env.fap_list[i]
    print(f.cache)
    # x = np.linspace(0,f.time_slot_delays.size+1,f.time_slot_delays.size)
    # y = f.time_slot_delays
    # x2 = np.linspace(0, f.time_out_hits.size + 1, f.time_out_hits.size)
    # y2 = f.time_out_hits
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x, y)
    # # plt.figure()
    # plt.subplot(1, 2, 2)
    # plt.plot(x2,y2)
# plt.show()

rewards = np.zeros(0)
for i in range(1,200):
    action = np.random.rand()*rl_env.fap_capacity + 1
    #rl_env.execute_frans_action(action)
    _, r = rl_env.step_by_frans(action)
    rewards = np.append(rewards,r)
    # for f in rl_env.fap_list.values():
    #     print("fap_id:", f.fap_id, " avg_delay: %f", rl_env.get_fap_avg_delay(f))
    print(r)
    print(rl_env.get_frans_avg_delay())
    print(rl_env.get_frans_avg_timeout())

    print(i)

x1, y1 = nrf.partition_avg(rl_env.time_slot_delays,40)


x2, y2 = nrf.partition_avg(rl_env.cache_hits, 40)


x3, y3 = nrf.partition_avg(rl_env.time_out_hits, 100)

x4, y4 = nrf.partition_avg(rewards, 1)

plt.figure()
plt.subplot(1, 4, 1)
plt.plot(x1, y1)
plt.subplot(1, 4, 2)
plt.plot(x2, y2)
plt.subplot(1, 4, 3)
plt.plot(x3, y3)
plt.subplot(1, 4, 4)
plt.plot(x4, y4)
plt.show()