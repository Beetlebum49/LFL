import numpy as np
from Env.fap import FAP
from scipy.stats import truncnorm
import utils.general_func as gfunc


class FRANs:
    """
    fap_cnt                 int     fap数量
    cluster_size            int     一个fap集群容纳的数量，规定是奇数
    content_cnt             int     内容数量
    fap_capacity            int     每个fap容量
    is_non_iid              bool    表明每个fap的内容流行度分布类型， True表示不是独立同分布的， False表示是独立同分布的
    skw_base                float   zipf分布的函数参数的基准值
    skw_sd                  float   zipf分布的函数参数的标准差，对于non_iid的数据集有意义
    skw_scale               float   zipf分布函数参数的范围
    plateau                 float   zipf分布平滑函数
    delay_threshold         tuple   延迟容忍度门限值元组，这里设定三个段位
    srv_avg_delay           tuple   不同服务类型的基准值（即正态分布的平均值）
    content_threshold_dic   dic     记录每个文件的门限值
    """

    def __init__(self, fap_cnt: int, cluster_size: int, content_cnt: int, fap_capacity: int,
                 is_non_iid: bool, skw_base: float, skw_sd: float, skw_scale:float, plateau: float, delay_threshold: tuple,
                 srv_avg_delay: tuple):
        self.fap_cnt = fap_cnt
        self.fap_list = dict()
        self.content_threshold_dic = dict()  # 记录每个文件的延迟敏感度
        self.fap_capacity = fap_capacity
        self.is_non_iid = is_non_iid
        self.skw_base = skw_base
        self.skw_sd = skw_sd  # 正态分布标准差
        self.skw_scale = skw_scale
        self.plateau = plateau
        self.delay_threshold = delay_threshold
        self.srv_avg_delay = srv_avg_delay
        self.delay_threshold_proportion = (0.15, 0.4)

        self.gen_faplist(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale, plateau)
        self.set_delay_threshold()

    def gen_faplist(self, fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale, plateau):

        # 生成fap列表
        if not is_non_iid:
            for i in range(1, fap_cnt + 1):
                fap = FAP(i, fap_capacity, skw_base, plateau, content_cnt)
                self.fap_list[i] = fap
        else:
            if skw_scale/2 >= skw_base:
                print("invalid skw_scale: ", skw_scale)
                return
            # 基于正态分布生成zipf分布的参数
            skw_gen = gfunc.get_truncated_normal(mean=skw_base, sd=skw_sd, low=-skw_scale/2+skw_base, upp=skw_scale/2+skw_base )
            for i in range(1, fap_cnt + 1):
                skw_factor = skw_gen.rvs()
                fap = FAP(i, fap_capacity, skw_factor, plateau, content_cnt)
                self.fap_list[i] = fap
        self.gen_cluster(cluster_size)

    def gen_cluster(self, cluster_size):
        if cluster_size < 2 or cluster_size > self.fap_cnt or cluster_size % 2 == 0 :
            print("invalid cluster_size: ", cluster_size, "\n")
            return
        nxt, pre = int((cluster_size - 1) / 2), int((cluster_size - 1) / 2)
        for i in self.fap_list:
            for j in range(1, nxt + 1):
                self.fap_list[i].add_co_fap(self.fap_list[(i + j + self.fap_cnt - 1) % self.fap_cnt + 1])
            for j in range(1, pre + 1):
                self.fap_list[i].add_co_fap(self.fap_list[(i - j + self.fap_cnt - 1) % self.fap_cnt + 1])

    def set_delay_threshold(self):
        for i in range(1, self.fap_capacity + 1):
            a = np.random.rand()
            if a < self.delay_threshold_proportion[0]:
                self.content_threshold_dic[i] = self.delay_threshold[0]
            elif a < self.delay_threshold_proportion[1]:
                self.content_threshold_dic[i] = self.delay_threshold[1]
            else:
                self.content_threshold_dic[i] = self.delay_threshold[2]
