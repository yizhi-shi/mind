#!/usr/bin/env python3
"""
联想记忆（轨道B）— 翻书检索 + 外部书籍存储
============================================
基于 Willshaw 网络 + Sparse Distributed Memory 混合架构。
所有输入内容被持久化存储为"书籍条目"，
系统可以随时通过当前内部状态检索。

算法来源：
  - Willshaw, 1969 "Non-linear network for pattern storage and retrieval"
  - Kanerva, 1988 "Sparse Distributed Memory"
  - Hopfield, 1982 "Neural networks and physical systems with emergent collective computational abilities"
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque


class AssociativeMemory:
    """
    联想记忆系统（轨道B）
    
    功能:
      1. 存储：将输入的模式二值化 → 存入稀疏矩阵或独立槽
      2. 检索：用当前状态作为检索信号 → 返回最匹配的记忆
      3. 书籍模式：保持原始内容，支持"翻书"检索
    
    参数:
        n_neurons: 与CTRNN的神经元数量一致
        max_slots: 最大记忆槽数量（防止无限膨胀）
        sparsity: 记忆编码的稀疏度
    """
    
    def __init__(self, 
                 n_neurons: int = 1000,
                 max_slots: int = 10000,
                 sparsity: float = 0.05):
        
        self.n = n_neurons
        self.max_slots = max_slots
        self.sparsity = sparsity
        
        # 轨道B-核心：Willshaw稀疏关联矩阵
        # n_neurons x n_neurons，逻辑OR叠加
        self.W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        
        # 轨道B-扩展：独立记忆槽（Kanerva SDM风格）
        self.slots = []  # 每个元素: {'pattern': np.array, 'content': str, 'time': str, 'source': str}
        
        # 书籍存储（原始内容，不压缩）
        self.books = []  # 每个元素: {'content': str, 'embedding': np.array, 'time': str, 'source': str, 'tags': []}
        
        # 最近的检索历史（避免重复检索同一个记忆）
        self.recall_history = deque(maxlen=50)
        
        # 检索统计
        self.recall_count = 0
        self.hit_count = 0
    
    def encode_pattern(self, data: np.ndarray) -> np.ndarray:
        """将浮点向量编码为二进制稀疏模式"""
        # 取绝对值，阈值化
        threshold = np.percentile(np.abs(data), 100 * (1 - self.sparsity))
        if threshold == 0:
            threshold = 0.01
        binary = (np.abs(data) > threshold).astype(np.float32)
        return binary
    
    def store_pattern(self, 
                      pattern: np.ndarray, 
                      content: str = "",
                      source: str = "experience"):
        """
        存储一个模式到联想记忆
        
        参数:
            pattern: 神经模式（CTRNN状态）
            content: 关联的原始内容（文本/描述）
            source: 来源标签
        """
        binary = self.encode_pattern(pattern)
        
        # 1. 写入Willshaw矩阵（叠加）
        self.W = np.maximum(self.W, np.outer(binary, binary))
        
        # 1b. 密度控制——避免Willshaw矩阵过饱和
        current_density = np.mean(self.W > 0)
        if current_density > 0.2:
            # 超过20%密度 → 衰减矩阵，用遗忘系数
            self.W *= 0.9995
            # 再加一层稀疏化
            threshold = np.percentile(self.W, 98)  # 只保留Top 2%最强连接
            self.W[self.W < threshold] = 0
            self.W[self.W >= threshold] = 1.0  # 二值化
        
        # 2. 写入独立记忆槽
        slot = {
            'pattern': binary.copy(),
            'content': content,
            'time': datetime.now().isoformat(),
            'source': source
        }
        self.slots.append(slot)
        
        # 3. 如果超过最大槽数，移除最旧的
        if len(self.slots) > self.max_slots:
            self.slots.pop(0)
        
        return True
    
    def recall(self, 
               cue: np.ndarray, 
               n_results: int = 3,
               min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        用检索信号cue从联想记忆中检索
        
        参数:
            cue: 检索信号（当前CTRNN状态）
            n_results: 返回的最匹配数量
            min_similarity: 最低相似度阈值
        
        返回:
            按匹配度排序的记忆列表
        """
        self.recall_count += 1
        binary_cue = self.encode_pattern(cue)
        
        results = []
        
        # 方法1：Willshaw直接检索
        # W @ cue → 最活跃的记忆
        willshaw_activation = self.W @ binary_cue
        willshaw_similarity = np.mean(willshaw_activation > 0.5)
        
        # 方法2：遍历独立记忆槽（更精确）
        for i, slot in enumerate(self.slots):
            # 计算Hamming相似度
            match = np.mean(slot['pattern'] == binary_cue)
            
            if match > min_similarity:
                results.append({
                    'slot_id': i,
                    'pattern': slot['pattern'],
                    'content': slot['content'],
                    'similarity': float(match),
                    'time': slot['time'],
                    'source': slot['source']
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:n_results]
        
        if top_results:
            self.hit_count += 1
        
        # 记录到检索历史
        self.recall_history.append({
            'cue_snapshot': binary_cue[:100],  # 只存前100维做近似
            'top_result': top_results[0]['content'][:50] if top_results else None,
            'n_results': len(results),
            'time': datetime.now().isoformat()
        })
        
        return top_results
    
    def store_book(self, 
                   content: str, 
                   embedding: Optional[np.ndarray] = None,
                   source: str = "user_input",
                   tags: Optional[List[str]] = None):
        """
        存储一本书籍条目（原始内容 + 嵌入）
        
        参数:
            content: 原始文本内容
            embedding: 可选的嵌入向量（如果不提供，自动生成）
            source: 来源
            tags: 标签列表
        """
        if embedding is None:
            # 从内容生成简单的嵌入（字符哈希）
            np.random.seed(hash(content) % (2**31))
            embedding = np.random.randn(self.n) * 0.1
            np.random.seed()
        
        self.books.append({
            'content': content,
            'embedding': embedding,
            'time': datetime.now().isoformat(),
            'source': source,
            'tags': tags or []
        })
        
        # 同时也把嵌入存储为联想模式
        self.store_pattern(embedding, content[:200], source=f"book_{source}")
        
        return True
    
    def search_books(self, 
                     cue: np.ndarray, 
                     n_results: int = 3) -> List[Dict[str, Any]]:
        """
        在书籍存储中检索（"翻书"操作）
        
        参数:
            cue: 检索信号
            n_results: 返回最多匹配数
        
        返回:
            按相关性排序的书籍条目
        """
        binary_cue = self.encode_pattern(cue)
        
        scored = []
        for i, book in enumerate(self.books):
            # 计算cue与书籍嵌入的相似度
            book_binary = self.encode_pattern(book['embedding'])
            sim = np.mean(book_binary == binary_cue)
            
            # 也做文本关键词匹配（如果书籍有内容）
            if book['content']:
                # 从cue中提取"关键词"（通过激活模式的前几个显著维度）
                active_dims = np.argsort(np.abs(cue))[-10:]
                # 这个简化的关键词提取后面可以改进
                keyword_bonus = 0.0
            
            scored.append((sim, i, book))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [{
            'book_id': i,
            'content': book['content'],
            'similarity': sim,
            'time': book['time'],
            'source': book['source'],
            'tags': book['tags']
        } for sim, i, book in scored[:n_results] if sim > 0.05]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        return {
            'n_slots': len(self.slots),
            'n_books': len(self.books),
            'willshaw_density': float(np.mean(self.W > 0)),
            'recall_count': self.recall_count,
            'hit_count': self.hit_count,
            'hit_rate': self.hit_count / max(1, self.recall_count)
        }
    
    def save(self, path: str, path_books: Optional[str] = None):
        """保存记忆状态"""
        data = {
            'n': self.n,
            'max_slots': self.max_slots,
            'sparsity': self.sparsity,
            'W_rows': [],
            'W_cols': [],
            'W_vals': [],
            'slots': [{
                'pattern': s['pattern'].tolist(),
                'content': s['content'],
                'time': s['time'],
                'source': s['source']
            } for s in self.slots[-1000:]],  # 只保留最近1000条
            'recall_count': self.recall_count,
            'hit_count': self.hit_count
        }
        
        # 保存Willshaw矩阵的非零元素
        rows, cols = np.where(self.W > 0)
        data['W_rows'] = rows.tolist()
        data['W_cols'] = cols.tolist()
        data['W_vals'] = self.W[rows, cols].tolist()
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        
        # 单独保存书籍（可能会很大）
        if path_books:
            books_data = [{
                'content': b['content'],
                'embedding': b['embedding'].tolist(),
                'time': b['time'],
                'source': b['source'],
                'tags': b['tags']
            } for b in self.books]
            
            with open(path_books, 'w') as f:
                json.dump(books_data, f)
    
    def load(self, path: str, path_books: Optional[str] = None):
        """加载记忆状态"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.max_slots = data['max_slots']
        self.sparsity = data['sparsity']
        self.recall_count = data['recall_count']
        self.hit_count = data['hit_count']
        
        # 重建矩阵
        self.W = np.zeros((self.n, self.n), dtype=np.float32)
        for r, c, v in zip(data['W_rows'], data['W_cols'], data['W_vals']):
            self.W[r, c] = v
        
        # 重建记忆槽
        self.slots = []
        for s in data['slots']:
            self.slots.append({
                'pattern': np.array(s['pattern']),
                'content': s['content'],
                'time': s['time'],
                'source': s['source']
            })
        
        # 加载书籍
        if path_books and os.path.exists(path_books):
            with open(path_books, 'r') as f:
                books_data = json.load(f)
            self.books = [{
                'content': b['content'],
                'embedding': np.array(b['embedding']),
                'time': b['time'],
                'source': b['source'],
                'tags': b['tags']
            } for b in books_data]
        
        print(f"[AssociativeMemory] 已加载 {len(self.slots)} 个记忆槽, {len(self.books)} 本书")


if __name__ == '__main__':
    print("=== 联想记忆测试 ===")
    mem = AssociativeMemory(n_neurons=100, max_slots=100)
    
    # 测试1：存储与检索
    print("\n测试1：存储与检索")
    for i in range(5):
        pattern = np.random.randn(100)
        mem.store_pattern(pattern, f"记忆内容_{i}")
    
    cue = np.random.randn(100)
    results = mem.recall(cue, n_results=2)
    print(f"  检索结果: {len(results)} 条")
    for r in results:
        print(f"    相似度={r['similarity']:.3f}, 内容={r['content']}")
    
    # 测试2：书籍存储
    print("\n测试2：书籍存储与翻书")
    mem.store_book("从前有座山，山里有座庙。", source="test")
    mem.store_book("杭州的雨夜，一个外卖员蹲在保时捷的碎片前。", source="novel_ch1")
    
    book_results = mem.search_books(cue)
    print(f"  翻书结果: {len(book_results)} 条")
    for b in book_results:
        print(f"    相似度={b['similarity']:.3f}, 内容={b['content'][:50]}")
    
    # 测试3：保存加载
    print("\n测试3：保存与加载")
    mem.save('/tmp/test_memory.json', '/tmp/test_books.json')
    mem2 = AssociativeMemory(n_neurons=100, max_slots=100)
    mem2.load('/tmp/test_memory.json', '/tmp/test_books.json')
    print(f"  加载后槽数: {len(mem2.slots)}")
    print(f"  加载后书籍数: {len(mem2.books)}")
    print(f"  命中率: {mem2.hit_count}/{mem2.recall_count}")
    
    print("\n联想记忆测试完成 ✅")
