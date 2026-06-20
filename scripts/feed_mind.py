#!/usr/bin/env python3
"""
Mind 持续喂养脚本 — 定时输入训练数据
========================================
在不打扰用户的情况下，持续往Mind系统注入小说内容和对话，
让它积累"经历"并自我成长。

这个脚本由cronjob定时调用。
每次运行时，它会把《还剩7天》各章的文本片段循环喂给Mind，
让权重在一段时间内通过赫布+STDP学习到故事结构。

使用方式:
  python3 scripts/feed_mind.py
  
环境变量:
  MIND_API: Mind系统API地址（默认 http://127.0.0.1:8080）
  FEED_COUNT: 每次喂养的条数（默认 5）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import random
import json

API = os.environ.get('MIND_API', 'http://127.0.0.1:8080')
FEED_COUNT = int(os.environ.get('FEED_COUNT', '5'))


# ===== 训练数据池 =====
FEED_DATA = [
    # --- 小说设定 ---
    "沈一舟是个普通的外卖员，在杭州骑电动车送餐，月薪七千。",
    "林璐在萧山公司做行政，月薪五千五，怀孕十四周。医生说不能再打胎了。",
    "周建国开保时捷卡宴，是做湿地公园项目的甲方，很有钱。",
    "沈一舟的爸爸叫沈文礼，在弋阳当了三十二年小学语文老师，肺癌晚期。",
    "系统给沈一舟的任务：七天内被三万人记住，否则抹杀。",
    "沈一舟发现传播最快的是抖音视频，但需要让人记住他这个人而不是事件。",
    
    # --- 章节片段 ---
    "雨夜。萧山的一条路上。沈一舟的电动车前灯照见一辆黑色保时捷的时候已经来不及了。",
    "电动车的轮子擦过保时捷的车头灯，车身一歪，连人带车倒在地上。",
    "手机屏幕亮起来。一行字浮在通话记录的上面——【任务：使30000人记住你。时间：168:00:00。】",
    "沈一舟蹲在产检等候区的塑料椅上，发了五条抖音。旁边十四个孕妇都看了他一眼。",
    "他翻开那个笔记本。第一行写着：爸。沈文礼。",
    "今天记住了四十七个人。按这个速度，需要四百二十八天。",
    "沈一舟在工地门口蹲了三个小时。周建国出来的时候天已经快黑了。",
    "周建国扔给他一叠a4纸，说这些湿地点位你去跑一下，干完给两千。",
    "沈一舟在省环境监测中心的门口停下来。塑料袋里装着十七瓶水样。",
    "刘主任说初步结果：五个点位不合格，铅超七倍，氨氮超十二倍。",
    "周建国约他在万象城吃饭。沈一舟这辈子没进过那家日料店。",
    "晚上十点。万豪酒店419房间。沈一舟躺在床上盯着天花板。",
    "抖音播放量六十七万。记住人数两万七。还剩不到一天。",
    "爸说：我儿子出息了。沈一舟没说话，拿着手机的手在抖。",
    "倒数最后几分钟。三万人在新闻评论里打下了沈一舟的名字。",
    
    # --- 写作规则 ---
    "冷笔写深情，不说话写心动，写动作不写心理，让读者自己品。",
    "不用然后、此外、继而、即、则、便这些词。",
    "不许用瞳孔地震、嘴角上扬、深吸一口气、某种说不清的。",
    "对话写完之后自己读一遍，读起来不像人说话就要改。",
    "一章只做一件事——让读者更多了解角色，或者让故事往前走。",
]


def feed():
    """
    喂养Mind系统一次
    从FEED_DATA中随机选择FEED_COUNT条数据喂给它
    """
    selected = random.sample(FEED_DATA, min(FEED_COUNT, len(FEED_DATA)))
    
    results = []
    for i, text in enumerate(selected):
        try:
            resp = requests.post(
                f"{API}/api/chat",
                json={"text": text},
                timeout=10
            )
            data = resp.json()
            results.append({
                'input': text[:30],
                'status': 'ok',
                'response_len': len(data.get('response', ''))
            })
            
            # 每条之间间隔0.5秒，避免拥堵
            time.sleep(0.5)
            
        except Exception as e:
            results.append({
                'input': text[:30],
                'status': 'error',
                'error': str(e)
            })
    
    # 获取喂养后的状态
    try:
        state_resp = requests.get(f"{API}/api/state", timeout=5)
        state = state_resp.json()
    except:
        state = {}
    
    report = {
        'fed': len(results),
        'success': sum(1 for r in results if r['status'] == 'ok'),
        'errors': sum(1 for r in results if r['status'] == 'error'),
        'results': results,
        'system_state': {
            'steps': state.get('steps', '?'),
            'connections': state.get('connection_count', '?'),
            'memory_slots': state.get('memory_stats', {}).get('n_slots', '?'),
            'mean_activity': state.get('mean_activity', '?'),
        }
    }
    
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    report = feed()
    
    # 如果出错超过阈值，打印错误信息
    if report['errors'] > 0 and report['errors'] >= report['fed'] / 2:
        print(f"\n⚠️ 高失败率：{report['errors']}/{report['fed']} 条失败")
        print("检查Mind服务是否在运行。")
