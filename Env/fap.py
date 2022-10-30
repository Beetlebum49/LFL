import numpy as np
from numpy import random
import utils.general_func as gfunc


class FAP:
    def __init__(self, fap_id: int, capacity: int, skewness: float, plateau: float, content_cnt: int):
        self.fap_id = fap_id
        self.capacity = capacity
        self.skewness = skewness
        self.plateau = plateau
        self.cache = np.zeros(0,dtype=int)
        self.cache_set = set()  # 文件集合
        self.content_cnt = content_cnt
        self.co_faplist = list()
        # 0表示本地服务， 1表示协作， 2表示从中心获取
        self.srv_type = (0, 1, 2)

    def is_full(self):
        return self.cache.size == self.capacity

    def add_content(self, content_id):
        # 防御式编程，如果有了，就不需要加了
        if content_id in self.cache_set:
            return
        if self.cache.size == self.capacity:
            print("out of fap cache size")
            return
        self.cache = np.append(self.cache,content_id)
        self.cache_set.add(content_id)
        self.cache = np.sort(self.cache)

    def replace_content(self, action, content_id):
        # 防御式编程，如果有了，就不需要替换了
        if content_id in self.cache_set:
            return
        if self.cache.size < self.capacity:
            print("out of fap cache size")
            return
        action = int(action)
        if action == self.capacity:
            return
        oldContentId = self.cache[action]
        self.cache[action - 1] = content_id
        self.cache_set.add(content_id)
        self.cache_set.remove(oldContentId)
        self.cache = np.sort(self.cache)

    def get_request(self):
        req_contentId = gfunc.Mzipf(np.float64(self.skewness), np.float64(self.plateau), np.uint(1), np.uint(self.content_cnt))
        return req_contentId

    def add_co_fap(self, fap):
        self.co_faplist.append(fap)

    def srv_type(self, content_id):
        if self.is_local(content_id):
            return self.srv_type[0]
        elif self.is_neighbour(content_id):
            return self.srv_type[1]
        else:
            return self.srv_type[2]

    def is_local(self, content_id):
        if content_id in self.cache_set:
            return True
        else:
            return False

    def is_neighbour(self, content_id):
        if content_id in self.cache_set:
            return False
        for co_fap in self.co_faplist:
            if co_fap.is_local(content_id):
                return True
            else:
                return False

    def is_center(self, content_id):
        if (not self.is_local(content_id)) and (not self.is_neighbour(content_id)):
            return True
        return False

