import numpy as np
from LFL.Env.fap import FAP
from scipy.stats import truncnorm
import LFL.utils.general_func as gfunc


class FRANs:
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
    def __init__(self, fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, delay_threshold,  srv_avg_delay):
        self.fap_list = list()
        self.content_threshold_dic = dict() #记录每个文件的延迟敏感度
        self.is_non_iid = is_non_iid
        self.skw_base = skw_base
        self.skw_sd = skw_sd  # 正态分布标准差
        self.delay_threshold = delay_threshold
        self.srv_avg_delay = srv_avg_delay

        self.gen_faplist(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd)

    @staticmethod
    def gen_faplist(self, fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd):

        # 生成fap列表
        if not is_non_iid:
            for i in range(1, fap_cnt + 1):
                fap = FAP(i, fap_capacity, skw_base, content_cnt)
                self.fap_list.append(fap)
        else:
            # 基于正态分布生成zipf分布的参数
            skw_gen = gfunc.get_truncated_normal(mean=skw_base, sd=skw_sd, low=1, upp=skw_base * 2)
            for i in range(1, fap_cnt + 1):
                skw_factor = skw_gen.rvs()
                fap = FAP(i, fap_capacity, skw_factor, content_cnt)
                self.fap_list.append(fap)
        self.gen_cluster(cluster_size)

    @staticmethod
    def gen_cluster(self, cluster_size):
        if cluster_size < 2 or cluster_size or cluster_size % 2 == 0> self.fap_cnt:
            print("invalid cluster_size: ", cluster_size, "\n")
            return
        nxt, pre = int((cluster_size - 1) / 2), int((cluster_size - 1) / 2)
        for i in range(1, self.fap_cnt+1):
            for j in range(1, nxt+1):
                self.fap_list[i].add_co_fap(self.fap_list[(i + j + self.fap_cnt) % self.fap_cnt])
            for j in range(i, pre+1):
                self.fap_list[i].add_co_fap(self.fap_list[(i - j + self.fap_cnt) % self.fap_cnt])

    def
