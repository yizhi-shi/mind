#!/usr/bin/env bash
# 每5分钟运行一次
# 把Mind的喂养和睡眠操作放在一个脚本里

cd /home/agent/.hermes/profiles/novel_writer/mind

# 1. 喂养数据
python3 scripts/feed_mind.py 2>&1 | tail -3

# 2. 检查是否需要进行睡眠巩固
# 用/api/state判断是否运行超过6小时
STATE=$(curl -s http://127.0.0.1:8080/api/state 2>/dev/null)
if [ -n "$STATE" ]; then
    STEPS=$(echo "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin).get('steps',0))")
    MEMORY=$(echo "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin).get('memory_stats',{}).get('n_slots',0))")
    echo "[cron] 步数: $STEPS | 记忆槽: $MEMORY"
fi
