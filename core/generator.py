"""
自发文本生成模块 — 从神经活动状态生成文字流
================================================
使用CTRNN的活动模式直接驱动文字生成。
不是LLM——是状态→字符的映射。

核心思想：
  网络的活跃模式 = "想表达的东西"
  活跃神经元群 → 映射到字符/词汇表 → 生成文本流

算法来源：
  - Itti & Koch 显著性注意力模型
  - 迭代蒸馏（先骨架后细节）
"""

import numpy as np
import random
from typing import Optional, List, Tuple

class SpontaneousGenerator:
    """
    自发文本生成器
    
    从CTRNN的当前活动状态，自发产生文字输出。
    
    工作方式：
      1. 从网络活动中提取"显著性区域"
      2. 显著性区域映射到词汇/短语
      3. 通过联想记忆中的内容做"语义塑形"
      4. 生成一句有意义的文字
    """
    
    def __init__(self, ctrnn, memory, seed=None):
        self.ctrnn = ctrnn
        self.memory = memory
        if seed:
            random.seed(seed)
        
        # 词汇池（从记忆和固定词库混合）
        self.word_pool = [
            '雨', '夜', '灯', '路', '车', '人', '风', '门', '窗', '街',
            '天', '水', '云', '海', '树', '山', '光', '影', '声', '色',
            '远', '近', '深', '浅', '快', '慢', '急', '缓', '冷', '暖',
            '红', '绿', '蓝', '白', '黑', '灰', '明', '暗', '清', '浊',
            '站', '走', '跑', '坐', '躺', '望', '听', '闻', '触', '想',
            '写', '说', '问', '答', '知', '觉', '记', '忘', '梦', '醒'
        ]
        
        # 句式模板（从轻到重，从短到长）
        self.templates_short = [
            "{w1}。",
            "{w1}了。",
            "{w1}的{w2}。",
            "{w1}和{w2}。",
            "在{w1}。"
        ]
        
        self.templates_medium = [
            "{w1}在{w2}。",
            "{w1}的{w2}里。",
            "有{w1}的{w2}。",
            "从{w1}到{w2}。",
            "{w1}着{w2}。",
            "那是{w1}的{w2}。",
        ]
        
        self.templates_long = [
            "{w1}在{w2}中{w3}。",
            "{w1}的{w2}像{w3}一样。",
            "从{w1}到{w2}再到{w3}。",
            "那个{w1}在{w2}里{w3}着。",
            "不是{w1}也不是{w2}——是{w3}。",
        ]
        
        self.last_output = ""
        self.iteration = 0
    
    def _extract_signature(self, state: np.ndarray) -> Tuple[List[int], float]:
        """
        从CTRNN状态中提取"显著性签名"
        
        返回:
            top_dims: 最活跃的维度索引
            energy: 总能量
        """
        # 取绝对值作为活跃度
        activity = np.abs(np.tanh(state))
        energy = float(np.sum(activity ** 2))
        
        # 取最显著的前N个维度
        n_significant = max(3, min(15, int(energy * 100)))
        top_dims = np.argsort(activity)[-n_significant:].tolist()
        
        return top_dims, energy
    
    def _dim_to_word(self, dim: int) -> str:
        """将神经维度映射到词汇"""
        # 固定词汇映射
        idx = dim % len(self.word_pool)
        return self.word_pool[idx]
    
    def generate(self) -> str:
        """
        生成一句自发文本
        
        返回:
            生成的文本
        """
        self.iteration += 1
        
        state = self.ctrnn.get_state()
        top_dims, energy = self._extract_signature(state)
        
        # 从记忆中获取"语义背景"
        memories = self.memory.recall(state, n_results=1)
        semantic_context = ""
        if memories and memories[0]['similarity'] > 0.3:
            semantic_context = memories[0]['content'][:20]
        
        # 根据能量选择句型复杂度
        if energy < 0.5:
            templates = self.templates_short
        elif energy < 2.0:
            templates = self.templates_medium
        else:
            templates = self.templates_long
        
        # 从显著性维度选择词汇
        words = [self._dim_to_word(d) for d in top_dims[:5]]
        words = list(set(words))  # 去重
        
        # 从语义背景提取一些词
        semantic_words = []
        if semantic_context:
            for ch in semantic_context:
                if '\u4e00' <= ch <= '\u9fff':
                    semantic_words.append(ch)
        
        # 合并词汇池
        pool = words[:]
        random.shuffle(semantic_words)
        pool.extend(semantic_words[:6])
        random.shuffle(pool)
        
        # 确保有足够的词
        while len(pool) < 3:
            pool.append(random.choice(self.word_pool))
        
        # 选择模板
        template = random.choice(templates)
        w_count = template.count('{w')
        
        # 填充模板
        try:
            if w_count == 1:
                result = template.format(w1=pool[0])
            elif w_count == 2:
                result = template.format(w1=pool[0], w2=pool[1])
            elif w_count == 3:
                result = template.format(w1=pool[0], w2=pool[1], w3=pool[2])
            else:
                result = pool[0] + "。"
        except (IndexError, KeyError):
            result = "。".join(pool[:3]) + "。"
        
        self.last_output = result
        return result
    
    def generate_with_context(self, external_cue: Optional[str] = None) -> str:
        """
        带外部线索的生成
        
        参数:
            external_cue: 外部提示（可选）
        
        返回:
            生成的文本
        """
        if external_cue:
            # 将外部线索注入网络
            cue_pattern = np.zeros(self.ctrnn.n)
            for i, ch in enumerate(external_cue[:30]):
                idx = hash(ch) % self.ctrnn.n
                cue_pattern[idx] += 0.3
            self.ctrnn.inject_pattern(cue_pattern, strength=0.3)
            
            # 运行几步让网络"吸收"
            for _ in range(10):
                self.ctrnn.forward(hebbian_lr=0.001)
        
        return self.generate()
    
    def generate_paragraph(self, sentences: int = 3) -> str:
        """生成一段文字（多句）"""
        paragraph = []
        for i in range(sentences):
            s = self.generate()
            # 确保不会完全重复
            if s != self.last_output:
                paragraph.append(s)
        return "".join(paragraph)


# 快速测试
if __name__ == '__main__':
    print("=== 自发文本生成模块测试 ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.ctrnn import CTRNN
    from memory.associative_memory import AssociativeMemory
    
    ctrnn = CTRNN(n_neurons=100, sparsity=0.1)
    mem = AssociativeMemory(n_neurons=100, max_slots=50)
    
    # 注入一些记忆
    mem.store_book("雨夜，沈一舟骑电动车在街上跑。", source="test")
    mem.store_book("路灯照着湿漉漉的路面。", source="test")
    
    gen = SpontaneousGenerator(ctrnn, mem)
    
    print("\n测试1：无输入自发生成")
    for i in range(5):
        text = gen.generate()
        print(f"  [{i}] {text}")
    
    print("\n测试2：带外部线索")
    text = gen.generate_with_context("雨夜")
    print(f"  {text}")
    
    print("\n测试3：段落生成")
    para = gen.generate_paragraph(3)
    print(f"  {para}")
    
    print("\n✅ 自发文本生成模块测试通过")
