import Env.frans as frans
import Env.fap as fap
import numpy as np
import math

'''
此处构建一个属于强化学习的环境体类RLEnv，继承自FRANs类
'''


class RLEnv(frans.FRANs):
    def __init__(self, fap_cnt: int, cluster_size: int, content_cnt: int, fap_capacity: int,
                 is_non_iid: bool, skw_base: float, skw_sd: float, skw_scale: float, plateau: float,
                 delay_threshold: tuple,
                 srv_avg_delay: tuple,
                 reward_para: float):
        super().__init__(fap_cnt, cluster_size, content_cnt, fap_capacity,
                         is_non_iid, skw_base, skw_sd, skw_scale, plateau, delay_threshold, srv_avg_delay)
        self.reward_para = reward_para
        self.cur_req_fap = self.get_req_fap()  # 这个变量仅仅对单个网络集中训练有意义，后续如果是多个分布式训练，就不一样了

    # 初始化全局的缓存内容（填充每个fap）
    def init_frans_cache(self):
        for f in self.fap_list.values():
            self.fullfill_fap_cache(f)

    ''' 动作函数----------------------------------------------------------------
        动作函数和奖励函数是搭配的，求解奖励函数的同时执行缓存替代操作
        execute_frans_action frans的维度执行缓存替换操作
        execute_fap_action fap的维度执行缓存替换操作
    '''

    def execute_frans_action(self, action):
        action_index = int(action)
        executor_fap = self.cur_req_fap
        executor_fap.replace_content(action_index, executor_fap.latest_req)

    def execute_fap_action(self, fap: fap.FAP, action):
        action_index = int(action)
        fap.replace_content(action_index, fap.latest_req)

    ''' 状态函数----------------------------------------------------------------
        get_cur_frans_state
        获取当前frans系统的状态。
        get_next_frans_next_state
        这个函数逻辑非常重，非理勿动。
        状态中包含"下一次"请求的内容，请求的fap和缓存的内容;
        这里的“下一次”的内容不是实际timeslot中的内容，而是下一次fap可以执行缓存操作
        的内容编号（也即如果请求的内容在本地有缓存，那就不把这个作为“下一次”的状态的内
        容，直接跳过直至到内容满足允许fap做缓存替换操作）
        值得注意的是，此函数的调用一般是是要在执行动作函数调用之后

        以下的工具函数在get_next_frans_state中为了实现上述逻辑都得到调用
        fullfill_fap_cache(self, fap: fap.FAP):
        get_next_replaceable_content_id(self, next_req_fap: fap.FAP):
        record_timeslot_delay_hit_timeout(self, fap: fap.FAP, content_id: int):
        record_action_delay(self, fap: fap.FAP, content_id: int):
        
        上述逻辑对于get_cur_fap_cluster_state和get_next_fap_cluster_state也是同理的

    '''

    def get_cur_frans_state(self):
        req_fap = self.cur_req_fap
        return self.get_frans_state(req_fap, req_fap.latest_req)

    def get_next_frans_state(self):
        next_req_fap = self.get_req_fap()
        if not next_req_fap.is_full():
            print("warning, requested fap is not full, id: ", next_req_fap.fap_id)
            self.fullfill_fap_cache(next_req_fap)
        # 定义下一个收到请求的fap和请求的文件编号
        next_req_content = self.get_next_replaceable_content_id(next_req_fap)
        # 记录下次请求的内容编号
        next_req_fap.latest_req = next_req_content
        self.cur_req_fap = next_req_fap
        return self.get_cur_frans_state()

    def get_cur_fap_cluster_state(self, fap: fap.FAP):
        return self.get_cluster_state(fap)

    def get_next_fap_cluster_state(self, fap: fap.FAP):
        if not fap.is_full():
            self.fullfill_fap_cache(fap)
        # 定义该fap请求下一个可以引发缓存替换的文件编号
        next_req_content = self.get_next_replaceable_content_id(fap)
        fap.latest_req = next_req_content
        return self.get_cur_fap_cluster_state(fap)

    ''' 奖励函数----------------------------------------------------------------
        设计多种奖励函数，不同的奖励函数的用type参数来控制,注意，执行奖励函数的时候并没有完全把
        状态切换到下一个，只是执行了缓存替代的操作
        get_frans_cur_reward: 获取全局frans的奖励值
    '''

    def get_frans_cur_reward(self, action, type: int = 0):
        cur_avg_delay = self.get_frans_avg_delay()
        cur_avg_time_out_prob = self.get_frans_avg_timeout()
        # 执行aciton
        self.execute_frans_action(action)
        next_avg_delay = self.get_frans_avg_delay()
        next_avg_time_out_prob = self.get_frans_avg_timeout()
        reward = 0
        if type == 0:
            reward = math.exp(-self.reward_para * next_avg_delay - (1 - self.reward_para) * next_avg_time_out_prob)
        if type == 1:
            reward = math.exp(-self.reward_para * next_avg_delay - (1 - self.reward_para) * next_avg_time_out_prob) - \
                     math.exp(-self.reward_para * cur_avg_delay - (1 - self.reward_para) * cur_avg_time_out_prob)
        return reward

    def get_fap_cluster_reward(self, fap: fap.FAP, action, type: int = 0):
        cur_avg_delay = self.get_cluster_avg_delay(fap)
        cur_avg_time_out_prob = self.get_cluster_avg_timeout(fap)
        self.execute_fap_action(fap, action)
        next_avg_delay = self.get_cluster_avg_delay(fap)
        next_avg_time_out_prob = self.get_cluster_avg_timeout(fap)
        reward = 0
        if type == 0:
            reward = math.exp(-self.reward_para * next_avg_delay - (1 - self.reward_para) * next_avg_time_out_prob)
        if type == 1:
            reward = math.exp(-self.reward_para * next_avg_delay - (1 - self.reward_para) * next_avg_time_out_prob) - \
                     math.exp(-self.reward_para * cur_avg_delay - (1 - self.reward_para) * cur_avg_time_out_prob)
        return reward

    '''step函数，输入为aciton，输出为一个下一个状态，奖励值----------------------------------
    step_by_frans: frans维度的执行下一步
    step_by_fap: fap维度的执行下一步
    
    '''

    def step_by_frans(self, action):
        if action < 1:
            action = 1
        if action > self.fap_capacity:
            action = self.fap_capacity
        reward = self.get_frans_cur_reward(action, 0)
        next_frans_state = self.get_next_frans_state()
        return next_frans_state, reward

    def step_by_fap(self, action, fap: fap.FAP):
        if action < 1:
            action = 1
        if action > self.fap_capacity:
            action = self.fap_capacity
        reward = self.get_fap_cluster_reward(fap, action, 0)
        next_fap_state = self.get_next_fap_cluster_state(fap)
        return next_fap_state, reward

    '''step-------------------------------------------------------------------------
        重置fap中每个cache的缓存和对应的请求，但是不记录对应数据（这种重置是为了训练而设置的，所以记录）
    '''
    def reset(self, conf):
        if conf[0] == 1:
            for f in self.fap_list.values():
                f.reset()
            self.cur_req_fap = self.get_req_fap()
        return self.get_cur_frans_state()

    '''工具函数-------------------------------------------------------------------------
    记录、填充，多个其它函数会调用下述函数
    '''
    # 填充某个fap缓存，并记录结果
    def fullfill_fap_cache(self, fap: fap.FAP):
        while not fap.is_full():
            contentId = self.get_req_content(fap)
            self.record_timeslot_delay_hit_timeout(fap, contentId)
            fap.add_content(contentId)

    # 获取下一个可以执行缓存替换的contendId（即不是本地缓存的）
    def get_next_replaceable_content_id(self, next_req_fap: fap.FAP):
        next_req_content = next_req_fap.get_request()
        self.record_timeslot_delay_hit_timeout(next_req_fap, next_req_content)
        # 如果是本地服务，不满足缓存替换的条件
        while next_req_fap.is_local(content_id=next_req_content):
            next_req_content = next_req_fap.get_request()
            self.record_timeslot_delay_hit_timeout(next_req_fap, next_req_content)
        # 达到了缓存替换的条件, 并且记录此次的delay
        self.record_action_delay(next_req_fap, next_req_content)
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
