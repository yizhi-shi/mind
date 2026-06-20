#!/usr/bin/env python3
"""
远程喂小说脚本——通过HTTP API把小说内容喂给在外面运行的Mind。

用法:
    python3 scripts/feed_book_remote.py --host 192.168.6.248 --port 5000 /path/to/小说.txt --author 作者名

原理：
    读取本地（或网络可访问路径）的小说文件，通过Mind的chat API逐段喂入。
    这样无需把小说文件拷贝到容器内，直接在容器内执行脚本即可。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
import json
import urllib.request
import urllib.error


def split_into_chapters(text: str) -> list:
    """按章节分割文本"""
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
    
    if len(result) <= 1:
        # 按段落分组
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        result = []
        for i in range(0, len(paras), 10):
            chunk = '\n'.join(paras[i:i+10])
            if chunk.strip():
                result.append(chunk)
    
    return [c for c in result if len(c) > 50]


def chunk_text(text: str, max_len: int = 800) -> list:
    """文本切块"""
    chunks = []
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


def call_mind_api(host: str, port: int, text: str) -> dict:
    """调用Mind的chat API"""
    url = f'http://{host}:{port}/api/chat'
    data = json.dumps({'text': text}).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, 
                                  headers={'Content-Type': 'application/json'},
                                  method='POST')
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'response': f'[ERROR: {e}]'}


def get_mind_state(host: str, port: int) -> dict:
    """获取Mind当前状态"""
    url = f'http://{host}:{port}/api/state'
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except:
        return {}


def main():
    parser = argparse.ArgumentParser(description='远程喂小说给Mind')
    parser.add_argument('book_path', help='小说文件路径')
    parser.add_argument('--author', default='未知', help='作者名')
    parser.add_argument('--host', default='192.168.6.248', help='Mind所在主机')
    parser.add_argument('--port', type=int, default=5000, help='Mind端口')
    parser.add_argument('--batch-size', type=int, default=5, help='每批喂多少段（默认5），越大越快')
    parser.add_argument('--dry-run', action='store_true', help='只统计不喂')
    args = parser.parse_args()

    if not os.path.exists(args.book_path):
        print(f"❌ 文件不存在: {args.book_path}")
        # 尝试从 /opt/data 找
        alt = f'/opt/data/root/hermes-web-ui/hermes_data/profiles/novel_writer/mind/{os.path.basename(args.book_path)}'
        if os.path.exists(alt):
            args.book_path = alt
            print(f"   已在备选路径找到: {alt}")
        else:
            return

    import chardet

    # 自动检测编码
    with open(args.book_path, 'rb') as f:
        raw = f.read(100000)
    enc_result = chardet.detect(raw)
    enc = enc_result['encoding'] if enc_result['encoding'] else 'utf-8'
    if enc.lower() in ['gb2312', 'gbk']:
        enc = 'gb18030'
    print(f"   检测编码: {enc} (置信度 {enc_result.get('confidence', 0):.0%})")

    with open(args.book_path, 'r', encoding=enc, errors='replace') as f:
        text = f.read()

    book_name = os.path.splitext(os.path.basename(args.book_path))[0]
    print(f"\n📖 正在喂小说: {book_name}")
    print(f"   作者: {args.author}")
    print(f"   总字数: {len(text)}")
    
    # 检查Mind状态
    state = get_mind_state(args.host, args.port)
    if state:
        print(f"   Mind状态: {state.get('steps', '?')}步, {state.get('connection_count', '?')}连接")
    else:
        print("   ⚠️ 无法获取Mind状态，确认Mind已启动")
        return
    
    # 分章
    chapters = split_into_chapters(text)
    print(f"   章节数: {len(chapters)}")
    
    # 切段
    all_chunks = []
    for ci, chapter in enumerate(chapters, 1):
        sub_chunks = chunk_text(chapter)
        for chunk in sub_chunks:
            all_chunks.append((ci, chunk))
    
    print(f"   总段落数: {len(all_chunks)}")
    
    if args.dry_run:
        print(f"\n📊 统计信息（dry-run）:")
        print(f"   平均每段字数: {sum(len(c) for _, c in all_chunks) // len(all_chunks)}")
        print(f"   第一段预览: {all_chunks[0][1][:60]}...")
        return
    
    print(f"\n[开始喂食]")
    
    # 先喂书的标题和作者（整体介绍）
    intro = f"《{book_name}》——{args.author}著。这是一本小说。"
    call_mind_api(args.host, args.port, intro)
    print(f"   📋 已注入书籍信息: {intro}")
    
    # 分批次喂
    batch_size = args.batch_size
    total = len(all_chunks)
    
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i+batch_size]
        
        for ci, chunk in batch:
            # 给每段加章节标记，帮助Mind理解上下文
            prefix = f"[第{ci}章] "
            feed_text = prefix + chunk + " [END]"
            
            resp = call_mind_api(args.host, args.port, feed_text)
            if '[ERROR' in resp.get('response', ''):
                print(f"   ❌ 错误: {resp['response']}")
                continue
        
        progress = min(i + batch_size, total)
        print(f"   进度: {progress}/{total} ({progress*100//total}%)")
        
        # 每50批等一秒，别把Mind撑爆
        if (i // batch_size) % 10 == 0:
            time.sleep(0.5)
    
    print(f"\n✅ 喂食完成！")
    print(f"   共喂入 {total} 个段落")
    
    # 最后问个问题验证
    print(f"\n[验证] 问问Mind小说内容...")
    q = f"我刚给你喂了《{book_name}》，你记住了吗？"
    resp = call_mind_api(args.host, args.port, q)
    print(f"   Q: {q}")
    print(f"   A: {resp.get('response', '')}")
    
    # 检查最终状态
    state = get_mind_state(args.host, args.port)
    if state:
        print(f"\n   Mind状态: {state.get('steps', '?')}步, {state.get('connection_count', '?')}连接, {state.get('memory_stats', {}).get('n_books', '?')}本书")


if __name__ == '__main__':
    main()
