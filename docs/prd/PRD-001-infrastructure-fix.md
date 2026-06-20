# Mind PRD-001 — 基础设施修复

## Goal

修复 Mind 项目迁移到新的 `~/.hermes/mind/` 目录后的路径硬编码问题，确保所有保存/加载路径正确指到 `~/.hermes/mind/data/`，Mind 能正常启动运行，不报路径不存在或 JSON 解析错误。

## Tech Stack

- Python 3.10+
- numpy
- Flask
- 所有已安装的依赖

## 需要修改的文件

### 1. `core/mind.py` — Mind 主控系统

**位置:** 第 185 行，`Mind.__init__`

**当前代码:**
```python
save_dir: str = '/home/agent/.hermes/profiles/novel_writer/mind/data'
```

**要求:**
- 将默认 `save_dir` 改为检测当前脚本所在目录下的 `data/` 子目录
- 即自动检测 `os.path.dirname(__file__)` 的上层目录 → `data/`
- 如果 `data/` 不存在则自动创建

**判断条件:**
```python
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
candidate = os.path.join(script_dir, 'data')
```

### 2. `run.py` — 启动入口

**位置:** 第 143-151 行，自动检测数据目录的逻辑

**当前代码已经用了正确模式**（从 `os.path.dirname(__file__)` 自动检测），但需要验证与 `mind.py` 的路径保持一致。

**要求:**
- 启动时不传 `--save-dir` 时，自动检测到的目录应与 `Mind.__init__` 中的默认目录一致
- 日志输出目录时显示绝对路径，便于调试

### 3. 无其他修改 — 两个文件够

## 不需要修改

以下文件不需要动：
- `core/ctrnn.py` — save/load 路径由外部传入
- `memory/associative_memory.py` — save/load 路径由外部传入
- `core/sleep.py` — 不涉及文件路径
- `core/generator.py` — 不涉及文件路径
- `scripts/feed_book.py` — 由 Mind 实例传入路径
- `scripts/feed_book_remote.py` — 走 HTTP API，不涉及本地路径

## 验收标准

- [x] Mind 在 `~/.hermes/mind/` 目录下能正常启动
- [x] 启动时打印正确的 data 目录（`~/.hermes/mind/data/`）
- [x] `save()` 成功，文件写入 `~/.hermes/mind/data/`
- [x] 重启后 `_try_load()` 成功加载之前保存的数据
- [x] 所有旧的 `novel_writer` 路径引用已消除
