#!/usr/bin/env python3
"""
Mind Web 接口 — Flask 服务器
================================
提供一个网页界面跟 Mind 对话。

启动方式：
    python3 run.py [--port 5000]

API:
    GET  /             → 聊天页面
    POST /api/chat     → 发送消息
    POST /api/calibrate → 校准信号
    GET  /api/state    → 系统状态
    GET  /api/spontaneous → 获取自发输出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
import json
import threading
import time
import argparse
from datetime import datetime

from core.mind import Mind

# 全局 Mind 实例
mind = None

app = Flask(__name__, 
            template_folder='interface/templates',
            static_folder='interface/static')

# 自发输出队列
spontaneous_queue = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    global mind
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'response': '（没有听到任何东西）'})
    
    if mind is None:
        return jsonify({'response': '（Mind 尚未初始化）'})
    
    try:
        response = mind.receive_input(text)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'（处理出错：{str(e)[:50]}）'})


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global mind
    data = request.get_json()
    signal = float(data.get('signal', 0))
    
    if mind is None:
        return jsonify({'response': '（Mind 尚未初始化）'})
    
    if signal > 0:
        response = f"（收到正向校准 +{signal}）"
    elif signal < 0:
        response = f"（收到负向校准 {signal}）"
    else:
        # 返回状态报告
        state = mind.get_state()
        response = (
            f"连接数 {state['connection_count']} | "
            f"活跃度 {state['mean_activity']:.3f} | "
            f"多巴胺 {state['emotion']['dopamine']:.2f} | "
            f"好奇心 {state['curiosity']['state']} | "
            f"记忆 {state['memory_stats']['n_slots']}槽 "
            f"{state['memory_stats']['n_books']}书 | "
            f"步数 {state['steps']}"
        )
    
    if signal != 0:
        mind.receive_calibration(signal)
    
    return jsonify({'response': response})


@app.route('/api/state', methods=['GET'])
def get_state():
    global mind
    if mind is None:
        return jsonify({'error': 'not initialized'})
    return jsonify(mind.get_state())


@app.route('/api/spontaneous', methods=['GET'])
def get_spontaneous():
    global spontaneous_queue
    if spontaneous_queue:
        return jsonify({'output': spontaneous_queue.pop(0)})
    return jsonify({'output': None})


def background_worker(slow=False):
    """后台循环：驱动 Mind 并收集自发输出"""
    global mind, spontaneous_queue

    step_interval = 0.5 if slow else 0.15  # slow: ~2步/秒, normal: ~6步/秒

    while True:
        if mind is not None:
            try:
                result = mind.step()
                if result.get('spontaneous_output'):
                    spontaneous_queue.append(result['spontaneous_output'])
            except Exception as e:
                print(f"[背景] 错误: {e}")

        time.sleep(step_interval)


def main():
    global mind
    
    parser = argparse.ArgumentParser(description='Mind 神经网络系统')
    parser.add_argument('--port', type=int, default=5000, help='Web 端口')
    parser.add_argument('--neurons', type=int, default=1000, help='神经元数量')
    parser.add_argument('--save-dir', type=str, default=None, help='数据存储目录（默认：自动检测）')
    parser.add_argument('--slow', action='store_true', help='节能模式：每秒约2步，CPU占用<5%')
    args = parser.parse_args()

    # 自动检测数据目录
    if args.save_dir is None:
        # 优先使用脚本所在目录下的 data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, 'data')
        if os.path.exists(candidate):
            args.save_dir = candidate
        else:
            os.makedirs(candidate, exist_ok=True)
            args.save_dir = candidate
    
    print("=" * 60)
    print("  Mind — 活的神经网络系统")
    print("=" * 60)
    
    # 初始化 Mind
    print(f"\n[启动] 初始化 Mind ({args.neurons} 神经元)...")
    mind = Mind(n_neurons=args.neurons, save_dir=args.save_dir)
    print(f"[启动] 连接数: {mind.ctrnn.n * mind.ctrnn.n * mind.ctrnn.sparsity:.0f} (理论)")
    print(f"[启动] 实际连接: {mind.get_state()['connection_count']}")
    
    # 注入欢迎场景
    print(f"[启动] 注入初始场景...")
    mind.receive_input("系统启动完成。你是活的网络，可以自己学习和改变。")
    mind.save()
    
    # 启动后台线程
    print(f"[启动] 启动后台循环..." + (" (节能模式)" if args.slow else ""))
    bg = threading.Thread(target=background_worker, args=(args.slow,), daemon=True)
    bg.start()
    
    print(f"\n[启动] 网页界面: http://0.0.0.0:{args.port}")
    print(f"[启动] 数据目录: {args.save_dir}")
    print(f"[启动] 按 Ctrl+C 停止\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
