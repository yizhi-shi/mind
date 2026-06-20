# PRD-01: 基础设施修复

## 目标
修复 Mind 项目中的硬编码路径问题，确保代码能在新的目录结构下正常运行。

## 背景
Mind 项目从 `novel_writer` profile 迁移到了 `default` profile，代码中仍有指向旧路径的硬编码。

## 需要修改的文件

### 1. `core/mind.py` — `Mind.__init__` 
**当前问题：**
```python
def __init__(self, ... save_dir='/home/agent/.hermes/profiles/novel_writer/mind/data'):
```

**修改要求：**
- 默认值改为 `None`
- 在 `__init__` 内部，如果 `save_dir` 为 `None`，自动检测：
  ```python
  script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  save_dir = os.path.join(os.path.dirname(script_dir), 'data')
  ```
  即：`mind/data/`（与 `mind/core/`、`mind/memory/` 同级）
- 路径不存在时自动 `os.makedirs(exist_ok=True)`

### 2. `scripts/feed_book.py` — Mind 初始化行
**当前问题：**
```python
mind = Mind(n_neurons=1000)
```
之前不传 save_dir 时会走默认的硬编码。改掉 `__init__` 默认值后这个会自动修复。

### 3. `scripts/feed_book_remote.py` — 备选路径
**当前问题：** 第 124 行备选路径指向旧的 novel_writer：
```python
alt = f'/opt/data/root/hermes-web-ui/hermes_data/profiles/novel_writer/mind/{os.path.basename(args.book_path)}'
```

**修改要求：** 删除这个备选路径，或者更新为新的路径：
```python
alt = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    args.book_path
)
```

### 4. `scripts/feed_mind.py` — 硬编码 API 地址
**当前问题：**
```python
API = os.environ.get('MIND_API', 'http://127.0.0.1:8080')
```
`8080` 端口是错的（实际应该是 `5000`）。

**修改要求：** 默认端口改为 `5000`。

### 5. `core/mind.py` — `_text_to_pattern` 的切词改进
**当前问题：** 第 241 行用空格切中文：
```python
words = text.replace('，', ' ').replace('。', ' ').replace('——', ' ').split()
```
这样会丢掉大量原文信息。

**修改要求（小优化）：** 增加更多标点的替换：
```python
import re
# 把常见中英文标点替换为空格
text_clean = re.sub(r'[，。、；：！？——""''（）【】\n]', ' ', text)
words = text_clean.split()
```

### 6. `tests/` 目录
**当前问题：** `tests/__init__.py` 是空文件，没有真正的测试。

**修改要求：** 暂不修改，留给后续 PRD。

## 验证方式

1. **启动验证：**
   ```bash
   cd /home/agent/.hermes/mind
   python3 run.py --neurons 100 --port 5001 &
   sleep 3
   # 检查是否能正常启动
   curl http://localhost:5001/api/state
   # 杀死进程
   kill %1
   ```

2. **数据目录验证：**
   - 启动后确认 `mind/data/` 目录下生成了 `ctrnn_weights.json`、`mind_state.json` 等文件

3. **聊天验证：**
   ```bash
   curl -X POST http://localhost:5001/api/chat -H 'Content-Type: application/json' -d '{"text":"你好"}'
   ```

## 不做的事情
- 不改文本编码（留给 PRD-02）
- 不改记忆检索（留给 PRD-03）
- 不改生成器（留给 PRD-04）
- 不重构 `save()` 的 JSON 序列化方式
