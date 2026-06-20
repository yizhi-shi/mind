#!/usr/bin/env python3
"""
预热脚本 — 给 Mind 系统注入初始记忆
=====================================
让系统在启动后先"读书"——把《还剩7天》的章节内容
以及一些小说写作的常识注入它的联想记忆，
这样你跟它对话时它不再是个空白大脑。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import requests
import json

API_BASE = "http://127.0.0.1:8080"

def chat(text):
    r = requests.post(f"{API_BASE}/api/chat", json={"text": text})
    return r.json().get('response', '')

def calibrate(signal):
    r = requests.post(f"{API_BASE}/api/calibrate", json={"signal": signal})
    return r.json().get('response', '')

print("=" * 60)
print("  Mind 预热 — 注入小说知识和初始记忆")
print("=" * 60)

# 1. 注入《还剩7天》的小说基本信息
print("\n📖 注入小说设定...")
novel_seed = [
    "我们写了一本小说叫《还剩7天》。主角沈一舟是个杭州的外卖员，28岁。",
    "沈一舟骑电动车撞了一辆保时捷卡宴，赔不起。被一个神秘系统绑定。",
    "系统的任务：7天内被30000人记住，否则抹杀。倒计时168:00:00。",
    "沈一舟的女朋友叫林璐，27岁，做行政工作，月薪5500。怀孕14周。",
    "沈一舟的爸爸叫沈文礼，在老家当过32年小学语文老师，肺癌晚期。",
    "小说的核心：一个底层外卖员在7天里被逼到绝路，为了活下去做了一些事。",
    "这本书的写法：冷笔写深情，不说话写心动，写动作不写心理，让读者自己品。",
    "小说的分类是都市生活，在番茄小说上发表。笔名是星空下影子。"
]

for seed in novel_seed:
    resp = chat(seed)
    print(f"  ✓ {seed[:30]}...")
    time.sleep(0.5)

print("\n📖 注入章节概要...")
chapter_summaries = [
    "第一章：雨夜萧山，沈一舟闯黄灯撞了周建国的保时捷。系统绑定，168:00:00开始倒计时。",
    "第二章：系统说要在7天内被30000人记住。沈一舟不相信，但倒计时在走。他试着发视频——只有7个赞。",
    "第三章：凌晨跑单想通了什么。发朋友圈、发抖音、去产检时刷存在。追悼会清单第一行：爸。",
    "第四章：记住了47个人。他发现要让人'记住'需要新方法——让撞车那个豪车车主传他。",
    "第五章：蹲周建国三个小时。周建国给了他17个湿地点位的资料。沈一舟说有个发小在环境监测中心。",
    "第六章：骑了140公里取17瓶水样。5个点位水质不达标——铅超7倍、氨氮超12倍。",
    "第七章：送检测中心。留了一瓶编号07的当对照。周建国开始认真了。",
    "第八章：检测结果5个超标。周建国约他见面，请吃了日料。住进五星酒店。明天第7天。",
    "第九章：公告发布。媒体跟进。记住人数涨到27000。爸在电话里说看到新闻了。时间还剩不到一天。",
    "第十章：倒计时归零前完成30000人任务。三个月后林璐生了个女儿。系统显示'任务完成'。他把手机关了。"
]

for ch in chapter_summaries:
    resp = chat(ch)
    print(f"  ✓ {ch[:25]}...")
    time.sleep(0.3)

print("\n📖 注入写作方法和SOUL.md风格...")
writing_seeds = [
    "写作的规则：不用'然而''此外''随即''则''便'这些词。日常说话怎么说就怎么写。",
    "不用'他顿了一下''瞳孔猛地收缩''心跳漏了一拍''嘴角微微上扬'——这些在网文里已经被人看腻了。",
    "一章只做一件事——让读者多了解角色一点点，或者让故事往前走一小步。"
]

for seed in writing_seeds:
    resp = chat(seed)
    print(f"  ✓ {seed[:30]}...")
    time.sleep(0.3)

print("\n📖 注入老板信息...")
about_you = [
    "老板的笔名是亦知，合作写作的AI助手叫小云。",
    "老板的写作风格：冷笔写深情，不说话写心动，写动作不写心理，让读者自己品。",
    "老板现在在用短篇小说做尝试，目标是走番茄小说短篇路线。",
    "老板对AI腔零容忍——不允许出现任何像AI写出来的句式。",
    "我们不写长篇了，现在专注短篇。"
]

for info in about_you:
    resp = chat(info)
    print(f"  ✓ {info[:30]}...")
    time.sleep(0.3)

# 保存
print("\n💾 保存预热后的记忆...")
r = requests.get(f"{API_BASE}/api/state")
state = r.json()
print(f"  记忆: {state['memory_stats']['n_slots']} 槽 / {state['memory_stats']['n_books']} 书")
print(f"  连接: {state['connection_count']}")

print("\n✅ 预热完成！")
print("现在可以用你的网页跟 Mind 聊天了:")
print(f"  → http://localhost:8080")
