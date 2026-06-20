#!/usr/bin/env python3
"""
Mind 主控系统 — 整合所有核心模块
===================================
把 CTRNN（基底活动）、联想记忆（轨道B）、
校准接口、好奇心状态机整合为一个可运行的系统。

优化：
  - 响应不再跳跳糖式拼接关键词
  - 只返回强匹配的记忆（相似度>0.4才说话）
  - 不匹配时返回"安静"状态，不强行回话
  - 喂书后可检索相关段落

系统启动后：
  1. 自动加载之前保存的权重（如果存在）
  2. 进入持续"活着"的状态（每秒2-6步）
  3. 接收输入 → 更新权重 → 产生输出 → 继续
  4. 检测异常状态（好奇心/无聊/认知危机）
  5. 所有输入自动存入轨道B（书籍）
  6. 定期保存权重和记忆
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import threading
import time
import queue
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

from core.ctrnn import CTRNN
from memory.associative_memory import AssociativeMemory
from core.sleep import SleepConsolidation
from core.generator import SpontaneousGenerator
from core.embedding import NgramEmbedder


# ---- 神经调制（情绪系统） ----
class Neuromodulator:
    """
    神经调制系统 — 情绪/情感的最底层

    三个全局调制信号:
      - 多巴胺 → 学习率调制（校准信号→预测误差）
      - 血清素 → 时间贴现/惯性
      - 去甲肾上腺素 → 感觉增益/警觉

    算法来源: Doya, 2000 "Meta-learning and neuromodulation"
    """

    def __init__(self):
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.noradrenaline = 0.5
        self.alpha = 0.05

    def update(self,
               prediction_error: float = 0.0,
               calibration_signal: float = 0.0,
               novelty: float = 0.0):
        self.noradrenaline += self.alpha * (prediction_error - self.noradrenaline)
        self.noradrenaline = np.clip(self.noradrenaline, 0.1, 1.0)

        if calibration_signal != 0:
            target_da = 0.5 + calibration_signal * 0.4
            self.dopamine += self.alpha * 2 * (target_da - self.dopamine)
        else:
            self.dopamine += self.alpha * 0.1 * (0.5 - self.dopamine)
        self.dopamine = np.clip(self.dopamine, 0.05, 1.0)

        if novelty > 0.5:
            self.serotonin += self.alpha * (0.2 - self.serotonin)
        else:
            self.serotonin += self.alpha * 0.1 * (0.5 - self.serotonin)
        self.serotonin = np.clip(self.serotonin, 0.1, 0.9)

    def get_effective_lr(self, base_lr: float = 0.01) -> float:
        return base_lr * (0.5 + self.dopamine)

    def get_state(self) -> Dict[str, float]:
        return {
            'dopamine': round(self.dopamine, 3),
            'serotonin': round(self.serotonin, 3),
            'noradrenaline': round(self.noradrenaline, 3),
            'effective_lr': round(self.get_effective_lr(), 5)
        }


# ---- 好奇心状态机 ----
class CuriosityStateMachine:
    """
    好奇心持久化状态机

    状态:
      0 = 静止（稳定环境）
      1 = 活跃（无新信息持续一段时间）
      2 = 探索中（基底主动偏离）
      3 = 探索疲劳（所有方向试过）
      4 = 休眠（等待新信号）
    """

    def __init__(self):
        self.state = 0
        self.state_time = 0.0
        self.exploration_log = []
        self.last_exploration_time = 0.0
        self.boredom_threshold = 30.0
        self.tried_patterns = []

    def update(self,
               delta_t: float,
               novelty_score: float,
               network_entropy: float,
               calibration_active: bool) -> tuple:
        self.state_time += delta_t
        trigger_exploration = False
        trigger_crisis = False

        if self.state == 0:
            if novelty_score < 0.2 and self.state_time > self.boredom_threshold:
                self.state = 1
                self.state_time = 0
                trigger_exploration = True
        elif self.state == 1:
            trigger_exploration = True
            self.state = 2
            self.state_time = 0
        elif self.state == 2:
            if novelty_score > 0.5:
                self.state = 0
                self.state_time = 0
            elif self.state_time > self.boredom_threshold * 2:
                self.state = 3
                self.state_time = 0
        elif self.state == 3:
            if self.state_time > self.boredom_threshold:
                self.state = 4
                self.state_time = 0
            elif novelty_score > 0.3:
                self.state = 2
                self.state_time = 0
        elif self.state == 4:
            if novelty_score > 0.4:
                self.state = 1
                self.state_time = 0
                trigger_exploration = True

        if self.state_time > self.boredom_threshold * 4 and not calibration_active:
            trigger_crisis = True

        return self.state, trigger_exploration, trigger_crisis

    def log_exploration(self, pattern_hash: str, result: str):
        self.tried_patterns.append({
            'pattern': pattern_hash,
            'result': result,
            'time': datetime.now().isoformat()
        })
        if len(self.tried_patterns) > 100:
            self.tried_patterns.pop(0)

    def get_state_name(self) -> str:
        names = ['静止', '活跃', '探索中', '探索疲劳', '休眠']
        return names[self.state] if self.state < len(names) else '未知'


# ---- Mind 主系统 ----
class Mind:
    """
    Mind 主系统

    整合所有模块，提供统一的接口：
      - receive_input(text): 接收文字输入
      - receive_calibration(signal): 接收校准信号
      - step(): 前进一步
      - get_state(): 获取当前状态
      - get_output(): 获取当前输出
    """

    def __init__(self,
                 n_neurons: int = 1000,
                 save_dir: str = None):

        # 自动检测 save_dir（默认放在 mind/data/，与 core/、memory/ 同级）
        if save_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))  # core/
            save_dir = os.path.join(os.path.dirname(script_dir), 'data')  # mind/data/

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 核心网络
        self.ctrnn = CTRNN(
            n_neurons=n_neurons,
            sparsity=0.05,
            noise_std=0.15,
            use_soc=True
        )

        # 文本嵌入（纯 numpy，无需外部依赖）
        self.embedder = NgramEmbedder(target_dim=n_neurons)

        # 联想记忆
        self.memory = AssociativeMemory(
            n_neurons=n_neurons,
            max_slots=10000
        )

        self.modulator = Neuromodulator()
        self.curiosity = CuriosityStateMachine()
        self.sleep_system = SleepConsolidation(self.ctrnn, self.memory)
        self.generator = SpontaneousGenerator(self.ctrnn, self.memory)

        self.text_dim = 64

        self.input_buffer = []
        self.output_buffer = []
        self.last_user_input = ""
        self.last_output = ""

        self.running = False
        self.steps_run = 0
        self.start_time = datetime.now()
        self.last_save_time = time.time()
        self.save_interval = 60

        self.spontaneous_output_enabled = True
        self.spontaneous_interval = 200

        self.is_booted = False

        # 对话历史（最近5轮）
        self.conversation_history = []

        self._try_load()

    def _text_to_pattern(self, text: str) -> np.ndarray:
        """将文本编码为 CTRNN 可接收的输入模式（n-gram 语义嵌入）"""
        return self.embedder.encode(text)

    def _pattern_to_text_rough(self, pattern: np.ndarray) -> str:
        """从CTRNN状态中读出文本（联想记忆匹配）"""
        memories = self.memory.recall(pattern, n_results=1)
        if memories and memories[0]['similarity'] > 0.3:
            return memories[0]['content']
        return ""

    def _is_same_question(self, a: str, b: str) -> bool:
        """判断两个输入是否问同一个意思"""
        if not a or not b:
            return False
        # 如果两者都有相同的核心词
        words_a = set(w for w in a if '\u4e00' <= w <= '\u9fff')
        words_b = set(w for w in b if '\u4e00' <= w <= '\u9fff')
        common = words_a & words_b
        if len(common) >= 2 and len(common) >= min(len(words_a), len(words_b)) * 0.5:
            return True
        return False

    def receive_input(self, text: str) -> str:
        """
        接收外部输入（文字）

        优化：不强行回话。强匹配才回答，否则报告"安静"状态。
        """
        self.last_user_input = text

        # 1. 编码
        input_pattern = self._text_to_pattern(text)

        # 2. 存储到书籍
        self.memory.store_book(
            content=text,
            embedding=input_pattern,
            source='user_input',
            tags=['user_input', datetime.now().strftime('%Y-%m-%d')]
        )

        # 3. 注入CTRNN，运行几步
        self.ctrnn.inject_pattern(input_pattern, strength=0.3)
        for i in range(30):
            state = self.ctrnn.forward(external_input=input_pattern * 0.1)
            self.steps_run += 1

        # 4. 检索强匹配的记忆
        memories = self.memory.recall(state, n_results=5)

        # 过滤：只保留高相似度的（>0.4）
        strong_memories = [m for m in memories if m.get('similarity', 0) > 0.4]

        # 5. 生成响应
        self.last_output = self._generate_response(text, strong_memories, state)
        self.output_buffer.append({
            'time': datetime.now().isoformat(),
            'input': text,
            'output': self.last_output,
            'state': self.get_state()
        })

        # 更新对话历史
        self.conversation_history.append({'role': 'user', 'content': text})
        self.conversation_history.append({'role': 'assistant', 'content': self.last_output})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return self.last_output

    def _generate_response(self, user_input: str, strong_memories: list, current_state: np.ndarray) -> str:
        """生成响应——强匹配才回答"""
        state = self.ctrnn.get_state()
        activity = self.ctrnn.get_activity()
        active_level = np.mean(activity)
        emotions = self.modulator.get_state()

        # 检查是否问的是同一个意思（防止重复回答）
        if self.conversation_history:
            last_user_msg = None
            for msg in reversed(self.conversation_history):
                if msg['role'] == 'user':
                    last_user_msg = msg['content']
                    break
            if last_user_msg and last_user_msg != user_input and self._is_same_question(user_input, last_user_msg):
                # 找到上次的回答
                last_assistant = None
                found = False
                for msg in reversed(self.conversation_history):
                    if msg['role'] == 'assistant':
                        if found:
                            last_assistant = msg['content']
                            break
                    elif msg['content'] == last_user_msg:
                        found = True
                if last_assistant:
                    return f"（和刚才的问题类似，我上次的回答是：{last_assistant[:80]}……）"

        # ---- 情况1：有强匹配记忆 ----
        if strong_memories:
            # 取最高相似度的记忆
            best = strong_memories[0]
            content = best.get('content', '')
            sim = best.get('similarity', 0)
            tags = best.get('tags', [])

            # 如果高相似度的是书籍片段 → 返回原文
            if 'book' in tags and sim > 0.5:
                # 取中间一段，避免太长
                if len(content) > 200:
                    # 找句子边界
                    content_short = content[:200]
                    last_period = max(content_short.rfind('。'), content_short.rfind('！'), content_short.rfind('？'))
                    if last_period > 50:
                        content_short = content_short[:last_period+1]
                    return f"（翻到一段相关的：）{content_short}"

            # 如果是用户之前输入的内容（关键词匹配强）
            if sim > 0.6 and len(content) < 150:
                return f"（这个我记得。{content}）"

            # 如果是高质量匹配但不长的
            if len(content) < 300:
                return f"（让我想想……{content}）"

            # 长内容，取关键部分
            return f"（我找到了一些相关的记忆，相似度{sim:.2f}）……"

        # ---- 情况2：没有强匹配，但对话历史有相关内容 ----
        if self.conversation_history:
            # 如果之前有关于类似主题的对话
            for msg in reversed(self.conversation_history):
                if msg['role'] == 'user' and msg['content'] != user_input:
                    if any(w in user_input for w in msg['content'] if '\u4e00' <= w <= '\u9fff'):
                        return f"（这和刚才的话题有关联，但我还在消化）"

        # ---- 情况3：完全新的内容——安静接受 ----
        return f"（安静地接收）"

    def receive_calibration(self, signal: float):
        self.modulator.update(
            calibration_signal=signal,
            prediction_error=abs(signal),
            novelty=0.5 if abs(signal) > 0.3 else 0.0
        )
        if signal < 0:
            self.ctrnn.growth_threshold *= 0.98
        elif signal > 0:
            self.ctrnn.growth_threshold *= 1.02

    def step(self) -> Dict[str, Any]:
        """主循环的一步"""
        self.steps_run += 1

        state = self.ctrnn.forward()
        activity = self.ctrnn.get_activity()
        active_level = np.mean(activity)
        entropy = self._compute_entropy(activity)

        self.modulator.update(
            prediction_error=abs(0.5 - active_level),
            novelty=entropy / 3.0 if entropy > 0 else 0.0
        )

        novelty_score = entropy / 3.0
        self.curiosity.update(
            delta_t=self.ctrnn.dt,
            novelty_score=min(1.0, novelty_score),
            network_entropy=entropy,
            calibration_active=False
        )

        # 睡眠巩固
        sleep_report = None
        if self.sleep_system.should_sleep(time.time(), idle_threshold=300):
            sleep_report = self.sleep_system.full_sleep()
            print(f"[睡眠] 周期#{sleep_report['sleep_cycle']}: "
                  f"NREM重放{sleep_report['nrem_replayed']}条 "
                  f"REM形成{sleep_report['rem_pairs_formed']}对")

        # 自动保存
        if time.time() - self.last_save_time > self.save_interval:
            self.save()
            self.last_save_time = time.time()

        # 自发输出
        spontaneous_output = None
        if (self.spontaneous_output_enabled and
            self.steps_run % self.spontaneous_interval == 0 and
            self.curiosity.state in [1, 2]):

            generated = self.generator.generate_with_context()
            if generated and len(generated) > 2:
                spontaneous_output = f"（自发想法）{generated}"

            if not spontaneous_output or len(spontaneous_output) < 10:
                memories = self.memory.recall(state, n_results=2)
                if memories and memories[0].get('similarity', 0) > 0.4:
                    spontaneous_output = f"（翻到一段旧内容：{memories[0]['content'][:40]}……）"

        return {
            'step': self.steps_run,
            'state_vector': state[:10].tolist(),
            'mean_activity': float(active_level),
            'entropy': float(entropy),
            'emotion': self.modulator.get_state(),
            'curiosity_state': self.curiosity.get_state_name(),
            'spontaneous_output': spontaneous_output,
            'sleep_report': sleep_report
        }

    def _compute_entropy(self, activity: np.ndarray) -> float:
        bins = np.linspace(0, 1, 20)
        hist, _ = np.histogram(activity, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            return float(-np.sum(hist * np.log(hist)))
        return 0.0

    def get_state(self) -> Dict[str, Any]:
        activity = self.ctrnn.get_activity()
        stats = self.memory.get_stats()
        return {
            'uptime': str(datetime.now() - self.start_time).split('.')[0],
            'steps': self.steps_run,
            'mean_activity': float(np.mean(activity)),
            'connection_count': int(np.count_nonzero(self.ctrnn.W)),
            'emotion': self.modulator.get_state(),
            'curiosity': {
                'state': self.curiosity.get_state_name(),
                'state_time': round(self.curiosity.state_time, 1)
            },
            'memory_stats': stats,
            'growth_threshold': round(self.ctrnn.growth_threshold, 3),
            'noise_std': round(self.ctrnn.noise_std, 3)
        }

    def save(self):
        self.ctrnn.save(os.path.join(self.save_dir, 'ctrnn_weights.json'))
        self.memory.save(
            os.path.join(self.save_dir, 'memory.json'),
            os.path.join(self.save_dir, 'books.json')
        )
        state_data = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'steps': self.steps_run,
            'emotion': self.modulator.get_state(),
            'curiosity_state': self.curiosity.state,
            'last_input': self.last_user_input,
            'last_output': self.last_output,
            'growth_threshold': self.ctrnn.growth_threshold,
            'noise_std': self.ctrnn.noise_std
        }
        with open(os.path.join(self.save_dir, 'mind_state.json'), 'w') as f:
            json.dump(state_data, f, indent=2)
        print(f"[Mind] 已保存（步数 {self.steps_run}）")

    def _try_load(self):
        ctrnn_path = os.path.join(self.save_dir, 'ctrnn_weights.json')
        memory_path = os.path.join(self.save_dir, 'memory.json')
        books_path = os.path.join(self.save_dir, 'books.json')
        state_path = os.path.join(self.save_dir, 'mind_state.json')

        if os.path.exists(ctrnn_path):
            try:
                self.ctrnn.load(ctrnn_path)
                print(f"[Mind] CTRNN 权重已加载")
            except Exception as e:
                print(f"[Mind] CTRNN 加载失败: {e}")

        if os.path.exists(memory_path):
            try:
                self.memory.load(memory_path, books_path if os.path.exists(books_path) else None)
                print(f"[Mind] 记忆已加载")
            except Exception as e:
                print(f"[Mind] 记忆加载失败: {e}")

        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state_data = json.load(f)
                self.steps_run = state_data.get('steps', 0)
                self.last_user_input = state_data.get('last_input', '')
                self.last_output = state_data.get('last_output', '')
                print(f"[Mind] 系统状态已恢复（步数 {self.steps_run}）")
            except Exception as e:
                print(f"[Mind] 状态加载失败: {e}")

        self.is_booted = True

    def run_background(self, steps_per_second: int = 5):
        def _loop():
            self.running = True
            interval = 1.0 / steps_per_second
            while self.running:
                self.step()
                time.sleep(interval)

        self._bg_thread = threading.Thread(target=_loop, daemon=True)
        self._bg_thread.start()

    def stop_background(self):
        self.running = False


if __name__ == '__main__':
    print("=== Mind 主系统测试 ===")
    mind = Mind(n_neurons=500, save_dir='/tmp/mind_test')

    print("\n测试1：接收输入")
    response = mind.receive_input("你好，我是老板。今天想写一个故事。")
    print(f"  输入: 你好，我是老板。")
    print(f"  响应: {response}")

    print("\n测试2：后台运行（20步）")
    for i in range(20):
        result = mind.step()
        if i % 5 == 0:
            print(f"  步 {i}: 活跃度={result['mean_activity']:.4f}, 情绪={result['emotion']['dopamine']:.2f}, 好奇心={result['curiosity_state']}")

    print("\n测试3：更多输入")
    resp2 = mind.receive_input("系统出bug了，外卖员写一半卡住了。")
    print(f"  响应: {resp2}")

    print("\n测试4：校准信号")
    mind.receive_calibration(-0.5)
    print(f"  校准后情绪: {mind.modulator.get_state()}")

    print("\n测试5：状态查看与保存")
    state = mind.get_state()
    print(f"  活跃度: {state['mean_activity']:.4f}")
    print(f"  连接数: {state['connection_count']}")
    print(f"  好奇心: {state['curiosity']}")
    mind.save()

    print("\n测试6：重新加载验证")
    mind2 = Mind(n_neurons=500, save_dir='/tmp/mind_test')
    state2 = mind2.get_state()
    print(f"  加载后步数: {state2['steps']}")
    print(f"  加载后连接数: {state2['connection_count']}")

    print("\nMind 测试完成 ✅")
