#!/usr/bin/env python3
"""
CTRNN — Continuous-Time Recurrent Neural Network
==================================================
基底活动层。没有输入时也自发放电（DMN模拟）。
所有神经元是稀疏连接的，权重实时更新。

算法来源:
  - Beer, 1995 "On the dynamics of small continuous-time recurrent neural networks"
  - Butz & van Ooyen, 2005 (活动依赖的拓扑可塑性)
  - 自组织临界性（SOC）调控

优化：Oja规则全向量化，减少Python循环，整体提速10-50倍。
"""

import numpy as np
import json
import os
from typing import Optional, Tuple, List

class CTRNN:
    """
    Continuous-Time Recurrent Neural Network

    参数:
        n_neurons: 神经元数量
        dt: 时间步长（秒）
        tau: 时间常数（决定"惯性"）
        sparsity: 连接稀疏度（0.05 = 5%）
        noise_std: 基底噪声标准差
        use_soc: 是否启用自组织临界性调控
    """

    def __init__(self,
                 n_neurons: int = 1000,
                 dt: float = 0.1,
                 tau: float = 1.0,
                 sparsity: float = 0.05,
                 noise_std: float = 0.1,
                 use_soc: bool = True):

        self.n = n_neurons
        self.dt = dt
        self.tau = tau
        self.sparsity = sparsity
        self.noise_std = noise_std
        self.use_soc = use_soc

        # 神经元状态（膜电位）
        self.y = np.random.randn(n_neurons) * 0.1

        # 连接权重矩阵（稀疏）
        self.W = self._init_weights()

        # 每个神经元的活跃度追踪（用于结构生长）
        self.activity_trace = np.zeros(n_neurons)
        self.long_term_activity = np.zeros(n_neurons)

        # 全局活动统计（用于SOC调控）
        self.activity_history = []

        # 生长参数
        self.growth_threshold = 0.3
        self.growth_rate = 0.001
        self.decay_rate = 0.9995

        # 神经元时间常数（可变的，允许个体差异）
        self.tau_i = np.ones(n_neurons) * tau * (1 + np.random.randn(n_neurons) * 0.1)
        self.tau_i = np.clip(self.tau_i, 0.1, 5.0)

        self.step_count = 0

        # ---- 呼吸振荡器 ----
        self.breathe_phase = 0.0
        self.breathe_freq = 0.03
        self.breathe_amp = 0.5
        self.breathe_offset = 0.0

    def _init_weights(self) -> np.ndarray:
        """初始化稀疏连接权重"""
        W = np.zeros((self.n, self.n))
        n_connections = int(self.n * self.n * self.sparsity)

        rows = np.random.randint(0, self.n, n_connections)
        cols = np.random.randint(0, self.n, n_connections)

        weights = np.random.randn(n_connections) * 0.05
        W[rows, cols] = weights

        np.fill_diagonal(W, 0)
        return W

    def forward(self,
                external_input: Optional[np.ndarray] = None,
                hebbian_lr: float = 0.01) -> np.ndarray:
        """
        一步前向传播（优化版——全向量化）。

        参数:
            external_input: 外部输入向量（长度=n，或None）
            hebbian_lr: 赫布学习率

        返回:
            更新后的神经元状态
        """
        self.step_count += 1

        # 呼吸振荡器
        self.breathe_phase += self.breathe_freq
        breathe_mod = 1.0 + np.sin(self.breathe_phase) * self.breathe_amp * 0.3
        breathe_mask = np.sin(np.arange(self.n) * 0.1 + self.breathe_phase) > 0.0

        # 基底噪声（受呼吸调制）
        noise_scale = self.noise_std * breathe_mod
        noise = np.random.randn(self.n) * noise_scale

        # 输入整合
        pre_act = np.tanh(self.y)
        if external_input is not None:
            drive = self.W @ pre_act + external_input + noise
        else:
            drive = self.W @ pre_act + noise

        # 呼吸偏置
        breathe_drive = np.where(breathe_mask, np.sin(self.breathe_phase) * 0.1, 0.0)
        drive = drive + breathe_drive

        # CTRNN 动态
        dy = (-self.y + drive) / self.tau_i
        self.y = self.y + dy * self.dt
        self.y = np.clip(self.y, -5.0, 5.0)

        # 赫布更新（向量化——每10步）
        if self.step_count % 10 == 0:
            self._hebbian_update_vec(hebbian_lr)

        # 活跃度追踪
        current_activity = np.abs(np.tanh(self.y))
        self.activity_trace = 0.9 * self.activity_trace + 0.1 * current_activity

        # 长时活跃度
        if self.step_count % 100 == 0:
            self.long_term_activity = 0.99 * self.long_term_activity + 0.01 * self.activity_trace

        # 结构可塑性（每500步）
        if self.step_count % 500 == 0:
            self._structural_plasticity()

        # SOC调控（每200步）
        if self.use_soc and self.step_count % 200 == 0:
            self._soc_regulate()

        return self.y

    def _hebbian_update_vec(self, lr: float):
        """
        全向量化的Oja规则 + STDP。
        不再遍历每一行——用矩阵运算。
        """
        post = np.tanh(self.y)  # (n,) 激活值
        pre = post

        # Oja规则向量化:
        # ΔW[i,j] = lr * (post[i] * pre[j] - W[i,j] * post[i]^2)
        # 但只对 W != 0 的位置操作

        mask = np.abs(self.W) > 1e-10
        if not np.any(mask):
            return

        # post[i] * pre[j] —— 外积
        hebb_term = np.outer(post, pre)  # (n, n)
        # Oja归一化项
        oja_term = self.W * (post ** 2)[:, np.newaxis]  # (n, n)

        delta = lr * (hebb_term - oja_term)
        # 只更新已有连接
        np.add.at(self.W, np.where(mask), delta[mask])

        # STDP（每50步做一次）
        if self.step_count % 50 == 0:
            self._stdp_update_vec()

    def _stdp_update_vec(self):
        """向量化STDP"""
        # 随机选~100个连接
        n_check = min(100, int(self.n * self.n * self.sparsity * 0.01))
        if n_check < 1:
            return

        # 只从已有连接中选
        nonzeros = np.argwhere(np.abs(self.W) > 1e-10)
        if len(nonzeros) == 0:
            return
        n_check = min(n_check, len(nonzeros))
        idx = np.random.choice(len(nonzeros), n_check, replace=False)
        rows = nonzeros[idx, 0]
        cols = nonzeros[idx, 1]

        pre_act = np.tanh(self.y[cols])
        post_act = np.tanh(self.y[rows])
        delta = 0.001 * (pre_act * post_act)
        self.W[rows, cols] += delta

    def _structural_plasticity(self):
        """结构可塑性：生长新连接 + 退化旧连接"""
        high_activity = np.where(self.long_term_activity > self.growth_threshold)[0]
        low_activity = np.where(self.long_term_activity < self.growth_threshold * 0.3)[0]

        # 在高度活跃的神经元之间创建新连接
        new_connections = 0
        for i in high_activity:
            j = np.random.randint(0, self.n)
            if j == i or abs(self.W[i, j]) > 1e-10:
                continue
            p = self.long_term_activity[i] * self.long_term_activity[j] * self.growth_rate * 100
            if np.random.random() < p:
                self.W[i, j] = np.random.randn() * 0.01
                new_connections += 1

        # 退化低活跃度的旧连接
        degraded = 0
        for i in low_activity:
            active_cols = np.where(np.abs(self.W[i]) > 1e-10)[0]
            for j in active_cols:
                self.W[i, j] *= self.decay_rate
                if abs(self.W[i, j]) < 1e-8:
                    self.W[i, j] = 0
                    degraded += 1

        if new_connections > 0 or degraded > 0:
            current_sparsity = np.count_nonzero(self.W) / (self.n * self.n)
            target = self.sparsity
            if current_sparsity > target * 1.5:
                excess = int((current_sparsity - target) * self.n * self.n)
                nonzeros = np.argwhere(np.abs(self.W) > 1e-10)
                if len(nonzeros) > 0:
                    to_kill = nonzeros[np.random.choice(len(nonzeros), min(excess, len(nonzeros)), replace=False)]
                    for r, c in to_kill:
                        self.W[r, c] = 0

    def _soc_regulate(self):
        """自组织临界性调控"""
        activity = np.tanh(self.y)
        bins = np.linspace(-1, 1, 20)
        hist, _ = np.histogram(activity, bins=bins, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log(hist))) if len(hist) > 0 else 0

        self.activity_history.append(entropy)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)

        if len(self.activity_history) >= 10:
            avg_entropy = np.mean(self.activity_history[-10:])
            target_entropy = np.log(20) * 0.7

            if avg_entropy < target_entropy * 0.8:
                self.growth_threshold *= 0.95
                self.noise_std = min(0.3, self.noise_std * 1.05)
            elif avg_entropy > target_entropy * 1.2:
                self.growth_threshold = min(1.0, self.growth_threshold * 1.05)
                self.noise_std = max(0.02, self.noise_std * 0.95)

    def get_state(self) -> np.ndarray:
        return self.y.copy()

    def get_activity(self) -> np.ndarray:
        return np.abs(np.tanh(self.y))

    def inject_pattern(self, pattern: np.ndarray, strength: float = 0.5):
        if len(pattern) != self.n:
            pattern = np.resize(pattern, self.n)
        self.y = self.y * (1 - strength) + pattern * strength

    def save(self, path: str):
        data = {
            'n': self.n,
            'dt': self.dt,
            'tau': self.tau,
            'sparsity': self.sparsity,
            'noise_std': self.noise_std,
            'use_soc': self.use_soc,
            'y': self.y.tolist(),
            'W_nonzero_rows': [],
            'W_nonzero_cols': [],
            'W_nonzero_vals': [],
            'activity_trace': self.activity_trace.tolist(),
            'long_term_activity': self.long_term_activity.tolist(),
            'tau_i': self.tau_i.tolist(),
            'growth_threshold': float(self.growth_threshold),
            'step_count': self.step_count
        }

        nonzeros = np.argwhere(np.abs(self.W) > 1e-10)
        for r, c in nonzeros:
            data['W_nonzero_rows'].append(int(r))
            data['W_nonzero_cols'].append(int(c))
            data['W_nonzero_vals'].append(float(self.W[r, c]))

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        self.n = data['n']
        self.dt = data['dt']
        self.tau = data['tau']
        self.sparsity = data['sparsity']
        self.noise_std = data['noise_std']
        self.use_soc = data['use_soc']
        self.y = np.array(data['y'])
        self.activity_trace = np.array(data['activity_trace'])
        self.long_term_activity = np.array(data['long_term_activity'])
        self.tau_i = np.array(data['tau_i'])
        self.growth_threshold = data['growth_threshold']
        self.step_count = data['step_count']

        self.W = np.zeros((self.n, self.n))
        for r, c, v in zip(data['W_nonzero_rows'], data['W_nonzero_cols'], data['W_nonzero_vals']):
            self.W[r, c] = v

        print(f"[CTRNN] 已加载 {len(data['W_nonzero_rows'])} 个连接，步数 {self.step_count}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == '__main__':
    print("=== CTRNN 测试 ===")
    net = CTRNN(n_neurons=100, sparsity=0.1, noise_std=0.05)

    print("测试1：无输入自发活动（100步）")
    for i in range(100):
        state = net.forward()
        if i % 20 == 0:
            activity_level = np.mean(np.abs(np.tanh(state)))
            print(f"  步 {i}: 平均活跃度={activity_level:.4f}")

    print("\n测试2：有输入驱动")
    for i in range(50):
        inp = np.random.randn(100) * 0.5
        state = net.forward(external_input=inp)
        if i % 10 == 0:
            activity_level = np.mean(np.abs(np.tanh(state)))
            print(f"  步 {i}: 平均活跃度={activity_level:.4f}")

    print("\n测试3：保存与加载")
    net.save('/tmp/test_ctrnn.json')
    net2 = CTRNN(n_neurons=100, sparsity=0.1)
    net2.load('/tmp/test_ctrnn.json')
    print(f"  加载后步数: {net2.step_count}")
    print(f"  连接数: {np.count_nonzero(net2.W)}")

    print("\nCTRNN 测试完成 ✅")
