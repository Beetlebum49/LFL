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
    content_threshold_dic   dic     记录每个文件的延迟门限值
    """

    def __init__(self, fap_cnt: int, cluster_size: int, content_cnt: int, fap_capacity: int,
                 is_non_iid: bool, skw_base: float, skw_sd: float, skw_scale: float, plateau: float,
                 delay_threshold: tuple,
                 srv_avg_delay: tuple):
        self.fap_cnt = fap_cnt
        self.fap_list = dict()
        self.content_delay_threshold_dic = dict()
        self.cluster_size = cluster_size
        self.content_cnt = content_cnt
        self.fap_capacity = fap_capacity
        self.is_non_iid = is_non_iid
        self.skw_base = skw_base
        self.skw_sd = skw_sd  # 流行度参数正态分布标准差
        self.skw_scale = skw_scale
        self.plateau = plateau
        self.delay_threshold = delay_threshold
        self.srv_avg_delay = srv_avg_delay

        # 以下后续可以考虑可以再函数入参中配置
        self.delay_threshold_proportion = (0.05, 0.1)
        self.srv_delay_scale = 0.2
        self.delay_sd = 1

        # 以下是需要调用函数初始化的属性
        self.gen_faplist(fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale,
                         plateau)
        self.set_contents_delay_threshold()
        self.timeout_probility_array = self.gen_timeout_probility_array(delay_threshold, srv_avg_delay,
                                                                        self.srv_delay_scale, self.delay_sd)

        '''-------------------------------------------------------------------------------------------------
              下面的几个元素的定义在类fap中也有类似的，这里主要收集的是系统整体的效果
              time_slot_delays
              action_time_slot_delays
              cache_hits
              time_out_hits

        '''
        # 记录每个time_slot的延迟矩阵
        self.time_slot_delays = np.zeros(0, dtype=float)
        # 记录每个action time slot的延迟矩阵（如果请求内容不存在fap的缓存中，在称该slot为action slot）
        self.action_time_slot_delays = np.zeros(0, dtype=float)
        # 记录每个time slot缓存是否命中,0为未命中，1为命中
        self.cache_hits = np.zeros(0, dtype=float)
        # 记录每个time slot缓存是否超时，0为未超时，1为超时
        self.time_out_hits = np.zeros(0, dtype=float)

    def gen_faplist(self, fap_cnt, cluster_size, content_cnt, fap_capacity, is_non_iid, skw_base, skw_sd, skw_scale,
                    plateau):

        # 生成fap列表
        if not is_non_iid:
            for i in range(1, fap_cnt + 1):
                fap = FAP(i, fap_capacity, skw_base, plateau, content_cnt)
                self.fap_list[i] = fap
        else:
            if skw_scale / 2 >= skw_base:
                print("invalid skw_scale: ", skw_scale)
                return
            # 基于正态分布生成zipf分布的参数
            skw_gen = gfunc.get_truncated_normal(mean=skw_base, sd=skw_sd, low=-skw_scale / 2 + skw_base,
                                                 upp=skw_scale / 2 + skw_base)
            for i in range(1, fap_cnt + 1):
                skw_factor = skw_gen.rvs()
                fap = FAP(i, fap_capacity, skw_factor, plateau, content_cnt)
                self.fap_list[i] = fap
        self.gen_cluster(cluster_size)

    def gen_cluster(self, cluster_size):
        if cluster_size < 2 or cluster_size > self.fap_cnt or cluster_size % 2 == 0:
            print("invalid cluster_size: ", cluster_size, "\n")
            return
        nxt, pre = int((cluster_size - 1) / 2), int((cluster_size - 1) / 2)
        for i in self.fap_list:
            for j in range(1, nxt + 1):
                self.fap_list[i].add_co_fap(self.fap_list[(i + j + self.fap_cnt - 1) % self.fap_cnt + 1])
            for j in range(1, pre + 1):
                self.fap_list[i].add_co_fap(self.fap_list[(i - j + self.fap_cnt - 1) % self.fap_cnt + 1])

    # 生成超时概率矩阵
    def gen_timeout_probility_array(self, delay_threshold: tuple, srv_avg_delay: tuple, scale: float, sd: float):
        timeout_probility_array = np.zeros(shape=(len(srv_avg_delay), len(delay_threshold)), dtype=float)
        # timeout_probility_array[i][j] 表示 srv_avg_delay 的第i种服务类型对应delay_threshold种第j种门限的超时概率
        for i in range(0, timeout_probility_array.shape[0]):
            for j in range(0, timeout_probility_array.shape[1]):
                mean = sum(srv_avg_delay[0:i + 1])
                timeout_probility = 1 - gfunc.get_truncated_normal_probility(mean, sd, mean - scale,
                                                                             mean + scale, delay_threshold[j])
                timeout_probility_array[i][j] = timeout_probility
        return timeout_probility_array

    # 随机的为内容分配延迟敏感度的值, 这个是每个内容的固有属性，可以是为环境的值
    def set_contents_delay_threshold(self):
        for i in range(1, self.content_cnt + 1):
            a = np.random.rand()
            if a < self.delay_threshold_proportion[0]:
                self.content_delay_threshold_dic[i] = self.delay_threshold[0]
            elif a < self.delay_threshold_proportion[1]:
                self.content_delay_threshold_dic[i] = self.delay_threshold[1]
            else:
                self.content_delay_threshold_dic[i] = self.delay_threshold[2]

    def get_exp_delay(self, srv_type: int):
        if srv_type == 0:
            return self.srv_avg_delay[0]
        if srv_type == 1:
            return self.srv_avg_delay[0] + self.srv_avg_delay[1]
        if srv_type == 2:
            return self.srv_avg_delay[0] + self.srv_avg_delay[1] + self.srv_avg_delay[2]

    def get_realtime_delay(self, srv_type: int):
        delay_gen = gfunc.get_truncated_normal(mean=self.srv_avg_delay[0],
                                               sd=self.delay_sd,
                                               low=self.srv_avg_delay[0] - self.srv_delay_scale,
                                               upp=self.srv_avg_delay[0] + self.srv_delay_scale)
        if srv_type == 0:
            return delay_gen.rvs()
        if srv_type == 1:
            return delay_gen.rvs() + self.srv_avg_delay[1]
        if srv_type == 2:
            return delay_gen.rvs() + self.srv_avg_delay[1] + self.srv_avg_delay[2]

    # 随机生成产生收到请求的FAP
    def get_req_fap(self):
        req_fap_id = np.random.randint(low=1, high=self.fap_cnt + 1, size=1, dtype=int)[0]
        return self.fap_list[req_fap_id]

    # 基于流行度生成对应fap收到的内容请求
    def get_req_content(self, fap: FAP):
        return fap.get_request()

    # 获取某个文件的延迟门限编号
    def get_content_delay_threshold_index(self, content_id: int):
        for index, value in enumerate(self.delay_threshold):
            if self.content_delay_threshold_dic[content_id] == value:
                return index
        return 2

    '''-------------------------------------------------------------------------------------------------
        下面定义frans中的状态函数
        get_fap_state： 单个fap的状态（不包含请求的fap编号和请求的内容编号）
        get_cluster_state： 一个fap所属的簇的状态（包含请求的fap编号和请求的内容编号）
        get_frans_state：全局的状态（包含请求的fap编号和请求的内容编号）
        关于状态函数的设计，请求fap编号和请求文件编号全程放在每一个状态的前两位，且如果对应的状态不是请求fap，则编号为0,请求内容编号也为0
    '''

    # 参数说明：fap是被查询的fap, req_fap是当前时隙收到内容请求的fap
    def get_fap_state(self, fap: FAP, req_fap: FAP, req_content_id: int):
        if fap == req_fap:
            state = np.array([fap.fap_id, req_content_id])
        else:
            state = np.zeros(2)
        state = np.append(state, np.pad(fap.cache, (0, self.fap_capacity - fap.cache.size), 'constant', constant_values=(0, 0)))
        return state

    def get_cluster_state(self, fap: FAP, req_fap: FAP, req_content_id: int):
        state = self.get_fap_state(fap, req_fap, req_content_id)
        for co_fap in fap.co_faplist:
            state = np.append(state, self.get_fap_state(co_fap, req_fap, req_content_id))

    def get_frans_state(self, req_fap: FAP, req_content_id: int):
        state = np.zeros(0)
        for fap in self.fap_list.values():
            state = np.append(state, self.get_fap_state(fap, req_fap, req_content_id))
        return state

    '''-------------------------------------------------------------------------------------------------
    下面定义frans中的延迟函数
    get_fap_avg_delay： 基于单个fap的维度的延迟的期望
    get_cluster_avg_delay： 基于一个fap所属的簇的延迟的期望
    get_frans_avg_delay：全局的延迟的期望
    '''
    def get_fap_avg_delay(self, fap: FAP):
        avg_delay = 0
        for i in range(1, self.content_cnt + 1):
            srv_type = fap.get_srv_type(i)
            avg_delay += self.get_exp_delay(srv_type) * fap.pops[i - 1]
        return avg_delay

    def get_cluster_avg_delay(self, fap: FAP):
        size = self.cluster_size
        avg_delay = 0
        avg_delay += 1 / size * self.get_fap_avg_delay(fap)
        for co_fap in fap.co_faplist:
            avg_delay += 1 / size * self.get_fap_avg_delay(co_fap)
        return avg_delay

    def get_frans_avg_delay(self):
        size = self.fap_cnt
        avg_delay = 0
        for fap in self.fap_list.values():
            avg_delay += 1 / size * self.get_fap_avg_delay(fap)
        return avg_delay

    '''-------------------------------------------------------------------------------------------------
       下面定义frans中的超时函数
       get_fap_avg_timeout： 基于单个fap的维度的请求超时的概率
       get_cluster_avg_timeout： 基于一个fap所属的簇的请求超时的概率期望
       get_frans_avg_timeout：全局的请求超时的概率期望
    '''
    def get_fap_avg_timeout(self, fap: FAP):
        timeout_prob = 0
        for i in range(1, self.content_cnt + 1):
            srv_type = fap.get_srv_type(i)
            delay_threshold_id = self.get_content_delay_threshold_index(i)
            timeout_prob += fap.pops[i - 1] * self.timeout_probility_array[srv_type][delay_threshold_id]
        return timeout_prob

    def get_cluster_avg_timeout(self, fap: FAP):
        size = self.cluster_size
        timeout_prob = 0
        timeout_prob += (1 / size) * self.get_fap_avg_timeout(fap)
        for co_fap in fap.co_faplist:
            timeout_prob += 1 / size * self.get_fap_avg_timeout(co_fap)
        return timeout_prob

    def get_frans_avg_timeout(self):
        size = self.fap_cnt
        timeout_prob = 0
        for fap in self.fap_list.values():
            timeout_prob += 1 / size * self.get_fap_avg_timeout(fap)
        return timeout_prob

    '''-------------------------------------------------------------------------------------------------'''
    def add_time_slot_delay(self, delay: float):
        self.time_slot_delays = np.append(self.time_slot_delays, delay)

    def add_action_time_slot_delay(self, delay: float):
        self.action_time_slot_delays = np.append(self.action_time_slot_delays, delay)

    def add_cache_hits(self, srv_type: int):
        if srv_type < 2:
            self.cache_hits = np.append(self.cache_hits, 1)
        else:
            self.cache_hits = np.append(self.cache_hits, 0)

    def add_time_out_hits(self, req_delay: float, time_out_threshold: float):
        if req_delay <= time_out_threshold:
            self.time_out_hits = np.append(self.time_out_hits, 0)
        else:
            self.time_out_hits = np.append(self.time_out_hits, 1)

