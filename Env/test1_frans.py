import numpy as np
from Env.fap import FAP
from Env.frans import FRANs
from scipy.stats import truncnorm
import utils.general_func as gfunc

"""
 fap_cnt                 int     fap数量
 cluster_size            int     一个fap集群容纳的数量，规定是奇数
 content_cnt             int     内容数量
 fap_capacity            int     每个fap容量
 is_non_iid              bool    表明每个fap的内容流行度分布类型， True表示不是独立同分布的， False表示是独立同分布的
 skw_base                float   zipf分布的函数参数的基准值
 skw_sd                  float   zipf分布的函数参数的标准差，对于non_iid的数据集有意义
 delay_threshold         tuple   延迟容忍度门限值元组，这里设定三个段位
 srv_avg_delay           tuple   不同服务类型的基准值（即正态分布的平均值）
 content_threshold_dic   dic     记录每个文件的门限值
 """
fap_cnt = 5
cluster_size = 3
content_cnt = 40000
fap_capacity = 40
is_non_iid = True
skw_base = 0.8
skw_sd = 1
skw_scale = 0.3
plateau = 0.1
delay_threshold = (0.8, 8.65, 10)
srv_avg_delay = (0.5, 0.3, 8)

frans = FRANs(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale, plateau,
              delay_threshold, srv_avg_delay)
print(frans.__dict__, "\n")
for f in frans.fap_list.values():
    print(f.fap_id, f.co_faplist, "\n")
    print(f.__dict__, "\n")
# for i in range(1,1000):
#     print(frans.fap_list[1].get_request())
step = 0
while not frans.fap_list[1].is_full():
    content = frans.fap_list[1].get_request()
    step += 1
    frans.fap_list[1].add_content(content)
print(step, frans.fap_list[1].cache)

fap = frans.get_req_fap()
content_id = frans.get_req_content(fap)

print(frans.get_universal_state(fap, content_id))

# 测试流行度
print(fap.pops)

# 测试延迟函数
print(frans.get_fap_avg_delay(fap))
print(frans.get_frans_avg_delay())

# 测试超时概率
print(frans.timeout_probility_array)
print(frans.get_fap_avg_timeout(fap))
print(frans.get_fap_avg_timeout(frans.fap_list[1]))



