#!/usr/bin/env python3
"""
喂小说到 Mind 的记忆中。

用法:
    python3 scripts/feed_book.py path/to/novel.txt [--author 作者名]

支持文本格式：
    - 纯文本（章/节用"第X章"或"***"分隔）
    - 每章不超过2000字
"""

import sys
import os
import re
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from core.mind import Mind
from pathlib import Path


def split_into_chapters(text: str) -> list:
    """把文本按章节分割"""
    # 匹配各种章节标题模式
    patterns = [
        r'第[一二三四五六七八九十百千万0-9]+[章节回]',
        r'Chapter\s*\d+',
        r'\*\*\*',
        r'--------',
    ]
    combined = '(' + '|'.join(patterns) + ')'
    
    chapters = re.split(combined, text, flags=re.MULTILINE)
    
    result = []
    current = ''
    for i, part in enumerate(chapters):
        if re.match(combined, part):
            if current.strip():
                result.append(current.strip())
            current = part + '\n'
        else:
            current += part
    
    if current.strip():
        result.append(current.strip())
    
    # 如果没有分章成功，按段落分组（每10段一组）
    if len(result) <= 1:
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        result = []
        for i in range(0, len(paras), 10):
            chunk = '\n'.join(paras[i:i+10])
            if chunk.strip():
                result.append(chunk)
    
    # 过滤太短的
    result = [c for c in result if len(c) > 50]
    
    return result


def chunk_text(text: str, max_len: int = 1500) -> list:
    """把大段文本切成小块，适配Mind的记忆槽"""
    chunks = []
    
    # 先按句号/问号/感叹号/换行分段
    sentences = re.split(r'([。！？\n])', text)
    
    current = ''
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        punct = sentences[i+1] if i+1 < len(sentences) else ''
        segment = s + punct
        
        if len(current) + len(segment) > max_len and current:
            chunks.append(current.strip())
            current = segment
        else:
            current += segment
    
    if current.strip():
        chunks.append(current.strip())
    
    return [c for c in chunks if len(c) > 20]


def extract_highlights(chapters: list, n: int = 5) -> list:
    """提取关键片段作为单独的记忆"""
    highlights = []
    for ch in chapters:
        # 找对话多的段落
        dialogues = re.findall(r'[「「『』""][^「「『』""]{10,200}[」」』]', ch)
        highlights.extend(dialogues)
    
    # 去重，排序
    highlights = list(set(highlights))
    highlights.sort(key=len, reverse=True)
    
    return highlights[:n]


def main():
    parser = argparse.ArgumentParser(description='喂小说给Mind')
    parser.add_argument('book_path', help='小说文件路径')
    parser.add_argument('--author', default='未知', help='作者名')
    parser.add_argument('--slow', action='store_true', help='慢速模式（每喂一章等一会儿）')
    args = parser.parse_args()
    
    if not os.path.exists(args.book_path):
        print(f"❌ 文件不存在: {args.book_path}")
        return
    
    # 读取文件
    with open(args.book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    book_name = os.path.splitext(os.path.basename(args.book_path))[0]
    print(f"\n📖 正在喂小说: {book_name}")
    print(f"   作者: {args.author}")
    print(f"   总字数: {len(text)}")
    
    # 初始化Mind
    print("\n[初始化 Mind]")
    mind = Mind(n_neurons=1000)
    print(f"   已加载 {mind.get_state()['memory_stats']['n_slots']} 个记忆槽")
    
    # 拆分章节
    chapters = split_into_chapters(text)
    print(f"   识别到 {len(chapters)} 个章节/段落")
    
    # 整本书作为"书籍"存储
    print(f"\n[存储整本书]")
    book_id = f"{book_name}_{int(time.time())}"
    mind.memory.store_book(
        content=text,
        embedding=mind._text_to_pattern(text[:500]),
        source=f'book:{book_name}',
        tags=['book', book_name, args.author]
    )
    print(f"   整书已存储为: {book_id}")
    
    # 逐章喂给Mind
    print(f"\n[逐章喂食]")
    for i, chapter in enumerate(chapters):
        short = chapter[:100].replace('\n', ' ').strip()
        print(f"   章节 {i+1}/{len(chapters)}: {short}...", end=' ')
        
        # 编码为神经模式注入
        input_pattern = mind._text_to_pattern(chapter)
        mind.ctrnn.inject_pattern(input_pattern, strength=0.3)
        
        # 存储各个段落作为独立的记忆
        sub_chunks = chunk_text(chapter)
        for chunk in sub_chunks:
            emb = mind._text_to_pattern(chunk)
            mind.memory.store_book(
                content=chunk,
                embedding=emb,
                source=f'book:{book_name}:ch{i+1}',
                tags=['book', book_name, args.author, f'ch{i+1}']
            )
        
        print(f"({len(sub_chunks)}段)")
        
        if args.slow and i % 5 == 0:
            print("   [休息一下，让Mind消化...]")
            for _ in range(50):
                mind.step()
    
    # 提取亮点并单独存储
    print(f"\n[提取关键词句]")
    highlights = extract_highlights(chapters)
    for h in highlights:
        emb = mind._text_to_pattern(h)
        mind.memory.store_book(
            content=h,
            embedding=emb,
            source=f'book:{book_name}:highlight',
            tags=['highlight', book_name]
        )
    print(f"   已存储 {len(highlights)} 个精彩片段")
    
    # 运行一些步让网络消化
    print(f"\n[消化中...]")
    for _ in range(200):
        mind.step()
    
    # 保存
    print(f"\n[保存]")
    mind.save()
    
    # 最终状态
    state = mind.get_state()
    print(f"\n✅ 完成!")
    print(f"   记忆槽: {state['memory_stats']['n_slots']}")
    print(f"   书籍数: {state['memory_stats']['n_books']}")
    print(f"   连接数: {state['connection_count']}")
    print(f"   步数: {state['steps']}")


if __name__ == '__main__':
    main()
