import numpy as np
from numpy import random
import utils.general_func as gfunc


class FAP:
    def __init__(self, fap_id: int, capacity: int, skewness: float, plateau: float, content_cnt: int):
        self.fap_id = fap_id
        self.capacity = capacity
        self.skewness = skewness
        self.plateau = plateau
        self.cache = np.zeros(0, dtype=int)
        self.delay = np.zeros(0, dtype=float)
        self.cache_set = set()  # 文件集合
        self.content_cnt = content_cnt
        self.co_faplist: list[FAP] = []
        # 0表示本地服务， 1表示协作， 2表示从中心获取
        self.srv_type = (0, 1, 2)
        self.pops = self.gen_pops(skewness, plateau, content_cnt)

        # 记录每个time_slot的延迟矩阵
        self.time_slot_delays = np.zeros(0, dtype=float)

        # 记录每个action time slot的延迟矩阵（如果请求内容不存在fap的缓存中，在称该slot为action slot）
        self.action_time_slot_delays = np.zeros(0, dtype=float)

        # 记录每个time slot缓存是否命中,0为未命中，1为命中
        self.cache_hits = np.zeros(0, dtype=float)

        # 记录每个time slot缓存是否超时，0为未超时，1为超时
        self.time_out_hits = np.zeros(0, dtype=float)

        # 记录该fap最近的一次收到的请求的content_id
        self.latest_req = self.get_request()

        # 记录每个fap初始化的值
        self.origional_paras = dict()
        self.set_original_paras()

    def is_full(self):
        return self.cache.size == self.capacity

    def add_content(self, content_id: int):
        # 如果有了，就不需要加了
        if content_id in self.cache_set:
            return
        if self.cache.size == self.capacity:
            print("out of fap cache size")
            return
        self.cache = np.append(self.cache, content_id)
        self.cache_set.add(content_id)
        self.cache = np.sort(self.cache)

    def replace_content(self, action, content_id: int):
        # 防御式编程，如果有了，就不需要替换了
        if content_id in self.cache_set:
            return
        if self.cache.size < self.capacity:
            print("cache is not full, need just add")
            self.add_content(content_id)
            return
        action = int(action)
        if action >= self.capacity:
            return
        oldContentId = self.cache[action - 1]
        self.cache[action - 1] = content_id
        self.cache_set.add(content_id)
        self.cache_set.remove(oldContentId)
        self.cache = np.sort(self.cache)

    def get_request(self):
        req_contentId = gfunc.Mzipf(np.float64(self.skewness), np.float64(self.plateau), np.uint(1),
                                    np.uint(self.content_cnt))
        return req_contentId

    def add_co_fap(self, fap):
        self.co_faplist.append(fap)

    def get_srv_type(self, content_id: int):
        if self.is_local(content_id):
            return self.srv_type[0]
        elif self.is_neighbour(content_id):
            return self.srv_type[1]
        else:
            return self.srv_type[2]

    def is_local(self, content_id: int):
        if content_id in self.cache_set:
            return True
        else:
            return False

    def is_neighbour(self, content_id: int):
        if content_id in self.cache_set:
            return False
        for co_fap in self.co_faplist:
            if co_fap.is_local(content_id):
                return True
            else:
                return False

    def is_center(self, content_id: int):
        if (not self.is_local(content_id)) and (not self.is_neighbour(content_id)):
            return True
        return False

    def gen_pops(self, skewness: float, plateau: float, content_cnt: int):
        return gfunc.Mzipf_pops(np.float64(skewness), np.float64(plateau), np.uint(1),
                                np.uint(content_cnt), None)

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

    def set_original_paras(self):
        self.origional_paras["capacity"] = self.capacity
        self.origional_paras["st_ep_para"] = 1.1
        self.origional_paras["rate_para"] = 0.6
        self.origional_paras["inc_proportion"] = 0.5
        self.origional_paras["modifier_type"] = 0

    # 重置缓存内容和latest_req
    def reset(self, conf):

        def set_fap(f: FAP, s_ep, cur_ep, rate_para, inc_proportion, tp):
            e_ep = int(s_ep * (1 + rate_para))
            if cur_ep <= s_ep or cur_ep > e_ep:
                return False
            final_size = f.origional_paras["capacity"] * (1 + inc_proportion)
            f.capacity = gfunc.reset_size(s_ep, e_ep, cur_ep, f.origional_paras["capacity"], final_size, tp)
            return True

        paras = self.origional_paras
        if conf[0] == 0 or conf[0] == 2:
            self.general_reset()
        if conf[0] == 1:  # linear1
            if not set_fap(self, int(conf[4] / conf[3] * paras["st_ep_para"]), conf[-1],
                           paras["rate_para"], paras["inc_proportion"], paras["modifier_type"]):
                self.general_reset()

    # CONF = [0, VAR, MAX_EPISODES, MAX_EP_STEPS, MEMORY_CAPACITY, 0]
    def general_reset(self):
        self.cache = np.zeros(0, dtype=int)
        self.cache_set.clear()
        while not self.is_full():
            req_content = self.get_request()
            self.add_content(req_content)
        self.latest_req = self.get_request()
