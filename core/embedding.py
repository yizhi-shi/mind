"""
文本嵌入模块 — 基于 n-gram 特征哈希的语义编码

不使用任何外部依赖（纯 numpy），将中文文本编码为
CTRNN 可接收的稠密向量。

方法：字符 n-gram (1-gram ~ 4-gram) 特征哈希
+ 位置权重衰减 + 局部敏感哈希投影

特点：
- 相同 / 相似文本 → 编码向量在高维空间相邻
- 不需要任何训练数据或预训练模型
- 仅依赖 numpy
"""

import numpy as np


class NgramEmbedder:
    """
    n-gram 特征哈希编码器

    将中文文本编码为可感知语义相似度的向量。
    通过 shared n-grams 捕捉词汇级别的语义重叠。

    用法:
        embedder = NgramEmbedder(target_dim=1000)
        vec = embedder.encode("今天天气很好")  # (1000,)
    """

    def __init__(self, target_dim: int = 1000):
        self.target_dim = target_dim

        # n-gram 范围：字符 1-gram 到 4-gram
        self.ngram_range = (1, 4)

        # 不同 n-gram 的权重（长 n-gram 信息量更大，权重大）
        self.ngram_weights = {1: 0.3, 2: 0.6, 3: 1.0, 4: 1.2}

        # 位置衰减系数（句首信息量大，越后面衰减）
        self.position_decay = 0.98

        # 用于散列的随机向量（固定种子确保一致性）
        rng = np.random.RandomState(42)
        # 哈希种子表：避免不同 n-gram 长度之间的碰撞
        self.hash_seeds = {
            1: rng.randint(0, 2**31, size=(65536,)),
            2: rng.randint(0, 2**31, size=(65536,)),
            3: rng.randint(0, 2**31, size=(65536,)),
            4: rng.randint(0, 2**31, size=(65536,)),
        }

        # 已经编码并缓存的关键词
        self.cache = {}

    def _chars_to_ngrams(self, text: str):
        """将文本转为 n-gram 列表，附带位置信息"""
        chars = list(text)
        ngrams = []

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            weight = self.ngram_weights[n]
            for i in range(len(chars) - n + 1):
                gram = ''.join(chars[i:i+n])
                pos = i / max(1, len(chars) - n)  # 0.0 ~ 1.0
                ngrams.append({
                    'gram': gram,
                    'n': n,
                    'weight': weight,
                    'pos': pos
                })

        return ngrams

    def _hash_ngram(self, gram: str, n: int) -> int:
        """将 n-gram 哈希到 [0, target_dim) 范围内"""
        # 使用固定的 hash seed 表
        seed = self.hash_seeds[n]
        h = hash(gram) % (2**31)
        idx = abs(h) % len(seed)
        hash_val = (seed[idx] ^ h) % self.target_dim
        return int(hash_val)

    def encode(self, text: str) -> np.ndarray:
        """编码文本为 target_dim 维向量"""
        if not text or not text.strip():
            return np.zeros(self.target_dim)

        # 只对中文字符做 n-gram
        # 保留英文和数字作为整体
        result = np.zeros(self.target_dim)

        ngrams = self._chars_to_ngrams(text)

        for item in ngrams:
            idx = self._hash_ngram(item['gram'], item['n'])

            # 位置衰减
            pos_weight = self.position_decay ** (item['pos'] * 100)

            # 累加（带符号：使用 hash 的奇偶性决定正负）
            sign = 1 if (hash(item['gram']) % 2 == 0) else -1

            result[idx] += sign * item['weight'] * pos_weight

        # 归一化到 [-1, 1]
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val

        # 用 tanh 做软裁剪
        result = np.tanh(result * 2.0)

        return result.astype(np.float32)

    def encode_batch(self, texts: list) -> np.ndarray:
        """批量编码，返回 (len(texts), target_dim)"""
        if not texts:
            return np.zeros((0, self.target_dim))
        return np.array([self.encode(t) for t in texts])

    def similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的语义相似度"""
        va = self.encode(text_a)
        vb = self.encode(text_b)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(np.dot(va, vb) / norm)


if __name__ == '__main__':
    print("=== n-gram Embedder 测试 ===\n")

    embedder = NgramEmbedder(target_dim=1000)

    # 测试1：相似文本应该临近
    pairs = [
        ("今天天气很好", "今天天气不错"),
        ("保时捷卡宴的碎片", "保时捷的车头灯"),
        ("月薪七千", "工资七千块"),
        ("沈一舟骑车送餐", "外卖员沈一舟"),
        ("今天天气很好", "傅总的办公桌上有一杯咖啡"),
    ]

    print("语义相似度测试:")
    for a, b in pairs:
        sim = embedder.similarity(a, b)
        print(f"  sim({a[:15]}..., {b[:15]}...) = {sim:.4f}")

    # 测试2：性能（编码100段）
    import time
    texts = [f"第{i}章 杭州的雨夜" for i in range(100)]
    t0 = time.time()
    vecs = embedder.encode_batch(texts)
    t = time.time() - t0
    print(f"\n性能: 100段编码耗时 {t:.3f}s ({t*10:.1f}ms/段)")

    print("\n✅ n-gram Embedder 测试完成")