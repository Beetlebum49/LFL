import Env.frans as frans
import Env.fap as fap
import numpy as np

'''
此处构建一个属于强化学习的环境体类RLEnv，继承自FRANs类
'''


class RLEnv(frans.FRANs):
    def __init__(self, fap_cnt: int, cluster_size: int, content_cnt: int, fap_capacity: int,
                 is_non_iid: bool, skw_base: float, skw_sd: float, skw_scale: float, plateau: float,
                 delay_threshold: tuple,
                 srv_avg_delay: tuple):
        super().__init__(fap_cnt, cluster_size, content_cnt, fap_capacity,
                         is_non_iid, skw_base, skw_sd, skw_scale, plateau, delay_threshold, srv_avg_delay)

        self.cur_req_fap = self.get_req_fap() # 这个变量仅仅对单个网络集中训练有意义，后续如果是多个分布式训练，就不一样了

    def get_cur_frans_state(self):
        req_fap = self.cur_req_fap
        return self.get_frans_state(req_fap, req_fap.latest_req)

    '''
    get_cur_frans_next_state
    这个函数逻辑非常重，非理勿动。
    状态中包含"下一次"请求的内容，请求的fap和缓存的内容;
    这里的“下一次”的内容不是实际timeslot中的内容，而是下一次fap可以执行缓存操作
    的内容编号（也即如果请求的内容在本地有缓存，那就不把这个作为“下一次”的状态的内
    容，直接跳过直至到内容满足允许fap做缓存替换操作）
    
    '''

    def get_cur_frans_next_state(self, action):
        # 首先进行缓存添加/替换


        # 再定义下一个收到请求的fap和请求的文件编号
        next_req_fap: fap.FAP = self.get_req_fap()
        if not next_req_fap.is_full():
            self.fullfill_fap_cache(next_req_fap)
        next_req_content = self.get_next_replaceable_content_id(next_req_fap)
        # 记录下次请求的内容编号
        next_req_fap.latest_req = next_req_content
        self.cur_req_fap = next_req_fap
        return self.get_cur_frans_state()


    # 填充某个fap缓存，并记录结果
    def fullfill_fap_cache(self, fap: fap.FAP):
        while not fap.is_full():
            contentId = self.get_req_content(fap)
            self.record_timeslot_delay_hit_timeout(fap, contentId)

    # 获取下一个可以执行缓存替换的contendId（即不是本地缓存的）
    def get_next_replaceable_content_id(self, next_req_fap: fap.FAP):
        next_req_content = next_req_fap.get_request()
        self.record_timeslot_delay_hit_timeout(next_req_content, next_req_content)
        # 如果是本地服务，不满足缓存替换的条件
        while next_req_fap.is_local(content_id=next_req_content):
            next_req_content = next_req_fap.get_request()
            self.record_timeslot_delay_hit_timeout(next_req_content, next_req_content)
        # 达到了缓存替换的条件
        self.record_action_delay(next_req_content, next_req_content)
        return next_req_content

    def record_timeslot_delay_hit_timeout(self, fap: fap.FAP, content_id: int):
        # 首先获取关键参数，服务类型
        srv_type = fap.get_srv_type(content_id)
        # 针对此次的请求，在timeslot层面要做各种记录(个体的和总体的都需要记录）
        real_delay = self.get_realtime_delay(srv_type)
        # 对应fap进行记录
        fap.add_time_slot_delay(real_delay)
        fap.add_cache_hits(srv_type)
        fap.add_time_out_hits(real_delay, self.content_delay_threshold_dic[content_id])
        # 对应frans进行记录
        self.add_time_slot_delay(real_delay)
        self.add_cache_hits(srv_type)
        self.add_time_out_hits(real_delay, self.content_delay_threshold_dic[content_id])

    def record_action_delay(self, fap: fap.FAP, content_id: int):
        # 首先获取关键参数，服务类型
        srv_type = fap.get_srv_type(content_id)
        # 针对此次的请求，在timeslot层面要做各种记录(个体的和总体的都需要记录）
        real_delay = self.get_realtime_delay(srv_type)
        fap.add_action_time_slot_delay(real_delay)
        self.add_action_time_slot_delay(real_delay)
