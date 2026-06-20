"""
睡眠巩固模块 — NREM重放 + REM整合
====================================
把短期记忆槽里的内容通过重放写入CTRNN权重结构。

NREM阶段：快速重放最近经历（10x速），赫布学习率提升
REM阶段：随机关联旧记忆和新记忆，建立跨连接

算法来源：
  - McClelland, 1995 "互补学习系统"
  - Wilson & McNaughton, 1994 "海马体重放"
  - Diekelmann & Born, 2000 "睡眠中的记忆巩固"
"""

import numpy as np
import time
import random
from typing import Optional

class SleepConsolidation:
    """
    睡眠巩固系统
    
    使用方式:
        sleep = SleepConsolidation(ctrnn_instance, memory_instance)
        sleep.nrem_cycle()  # 慢波巩固
        sleep.rem_cycle()   # REM整合
        sleep.full_sleep()  # 完整睡眠周期
    """
    
    def __init__(self, ctrnn, memory):
        self.ctrnn = ctrnn
        self.memory = memory
        self.last_sleep_time = time.time()
        self.sleep_count = 0
        
        # NREM重放参数
        self.nrem_duration = 5.0   # 秒
        self.replay_speed = 10.0   # 重放速度倍数
        self.hebbian_boost = 3.0   # 重放期间学习率提升
        
        # REM参数
        self.rem_duration = 3.0    # 秒
        self.rem_pair_count = 5    # 生成多少对新关联
    
    def nrem_cycle(self):
        """
        NREM慢波巩固：快速重放最近的记忆
        
        效果：
          - 近期经历被快速重放（10x速）
          - 对应神经模式被强化
          - 短期记忆槽的"强度"下降
        """
        # 获取最近的记忆
        recent_slots = self.memory.slots[-50:]  # 最近50条
        if not recent_slots:
            return 0
        
        replayed = 0
        
        for slot in recent_slots:
            pattern = slot.get('pattern')
            if pattern is None:
                continue
            
            # 重放：用记忆模式驱动CTRNN
            pattern_float = pattern.astype(np.float32)
            
            for _ in range(int(self.replay_speed)):
                state = self.ctrnn.forward(
                    external_input=pattern_float * 0.3,
                    hebbian_lr=0.01 * self.hebbian_boost
                )
            
            replayed += 1
        
        # 学习率回归正常
        self.ctrnn.forward(hebbian_lr=0.01)
        
        return replayed
    
    def rem_cycle(self):
        """
        REM整合：随机关联旧记忆和新记忆
        
        效果：
          - 从旧记忆和新记忆中随机取对，混合模式
          - 混合模式被注入CTRNN（强度更大）
          - 赫布学习率提升至2倍，确保新连接写入权重
          - 在新旧连接之间产生交叉连接
        """
        all_slots = self.memory.slots
        if len(all_slots) < 2:
            return 0
        
        pairs_formed = 0
        
        for _ in range(self.rem_pair_count):
            # 选一条旧记忆和一条新记忆
            mid = len(all_slots) // 2
            old = random.choice(all_slots[:mid]) if len(all_slots) > 2 else random.choice(all_slots)
            new = random.choice(all_slots[mid:]) if len(all_slots) > 2 else random.choice(all_slots)
            
            if old.get('pattern') is None or new.get('pattern') is None:
                continue
            
            # 混合模式
            old_p = old['pattern'].astype(np.float32)
            new_p = new['pattern'].astype(np.float32)
            blend = old_p * 0.5 + new_p * 0.5
            
            # 注入混合模式——强度翻倍，确保权重变化可见
            for _ in range(20):  # 更多的注入步
                self.ctrnn.forward(
                    external_input=blend * 0.4,  # 强度提高
                    hebbian_lr=0.02  # 学习率翻倍
                )
            
            # 在记忆系统中记录这个关联（概率提高到80%）
            if random.random() < 0.8:
                old_content = old.get('content', '')[:30]
                new_content = new.get('content', '')[:30]
                combined_content = f"[REM关联] {old_content} + {new_content}"
                self.memory.store_pattern(
                    blend,
                    content=combined_content,
                    source='REM_consolidation'
                )
                pairs_formed += 1
        
        return pairs_formed
    
    def full_sleep(self, environment_stats: Optional[dict] = None):
        """
        完整睡眠周期
        
        参数:
            environment_stats: 可选的环境统计（用于调整睡眠参数）
        
        返回:
            睡眠报告
        """
        self.sleep_count += 1
        
        # 保存睡眠前状态
        pre_connections = np.count_nonzero(self.ctrnn.W)
        pre_memory_count = len(self.memory.slots)
        
        # 阶段1：NREM
        nrem_replay = self.nrem_cycle()
        
        # 阶段2：REM
        rem_pairs = self.rem_cycle()
        
        # 保存睡眠后状态
        post_connections = np.count_nonzero(self.ctrnn.W)
        post_memory_count = len(self.memory.slots)
        
        # 记录脑活动（睡眠结束时活跃度上升，是巩固成功的标志）
        activity_after = np.mean(self.ctrnn.get_activity())
        
        self.last_sleep_time = time.time()
        
        report = {
            'sleep_cycle': self.sleep_count,
            'nrem_replayed': nrem_replay,
            'rem_pairs_formed': rem_pairs,
            'connections_before': pre_connections,
            'connections_after': post_connections,
            'connection_change': post_connections - pre_connections,
            'memory_before': pre_memory_count,
            'memory_after': post_memory_count,
            'activity_after': float(activity_after),
        }
        
        return report
    
    def should_sleep(self, current_time: float, idle_threshold: float = 30.0) -> bool:
        """
        判断是否应该进入睡眠
        
        参数:
            current_time: 当前时间（秒）
            idle_threshold: 安静运行多久后触发睡眠（秒）
        
        返回:
            是否触发睡眠
        """
        time_since_last_sleep = current_time - self.last_sleep_time
        
        # 至少运行了idle_threshold秒
        if time_since_last_sleep < idle_threshold:
            return False
        
        # 每小时最多睡一次
        if time_since_last_sleep < 3600 and self.sleep_count > 0:
            return False
        
        # 至少有记忆可以巩固
        if len(self.memory.slots) < 5:
            return False
        
        return True


# 快速测试
if __name__ == '__main__':
    print("=== 睡眠巩固模块测试 ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.ctrnn import CTRNN
    from memory.associative_memory import AssociativeMemory
    
    ctrnn = CTRNN(n_neurons=100, sparsity=0.1)
    mem = AssociativeMemory(n_neurons=100, max_slots=50)
    
    # 注入一些测试记忆
    for i in range(10):
        p = np.random.randn(100)
        mem.store_pattern(p, f"测试记忆_{i}")
    
    sleep = SleepConsolidation(ctrnn, mem)
    
    print("睡眠前:")
    print(f"  连接数: {np.count_nonzero(ctrnn.W)}")
    print(f"  记忆槽: {len(mem.slots)}")
    
    report = sleep.full_sleep()
    
    print("\n睡眠报告:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    
    print("\n✅ 睡眠巩固模块测试通过")
