# Second Brain — LLM 应用开发入门实验

从零手写一个具备 RAG、记忆持久化、意图路由和语义缓存能力的 AI 知识助手。不依赖 LangChain 等框架，每一层都从原理出发自己实现，通过实验验证理论。

## 项目结构

```
practice/
├── data/                    # 知识库文档（理论笔记 + 实验记录）
├── logs/                    # 对话测试历史
├── layer1/                  # 大模型 API 调用 + 多轮对话历史管理
├── layer2/                  # Embedding + 余弦相似度
├── layer3/                  # Chroma 向量数据库 + 文档导入
├── layer4/                  # 完整 RAG 流水线
├── layer5/                  # 记忆持久化（session 管理 + RAG 集成）
└── layer6/                  # Router 意图分类 + 语义缓存
```

## 七层架构

| 层 | 主题 | 核心文件 | 状态 |
|---|---|---|---|
| Layer 1 | 大模型 API 调用 + 多轮对话历史管理 | `layer1/chat_mine.py` | ✅ |
| Layer 2 | Embedding + 余弦相似度 | `layer2/embeddings.py` | ✅ |
| Layer 3 | Chroma 向量数据库 + 文档导入 | `layer3/knowledge_base.py`, `layer3/ingest.py` | ✅ |
| Layer 4 | 完整 RAG 流水线 | `layer4/rag.py`, `layer4/chat_rag.py` | ✅ |
| Layer 5 | 记忆与持久化 | `layer5/memory.py`, `layer5/chat_memory.py` | ✅ |
| Layer 6 | Router 意图分类 + 语义缓存 | `layer6/router.py`, `layer6/cache.py`, `layer6/chat_layer6.py` | ✅ |
| Layer 7 | 系统评估（RAGAS） | — | 🔜 |

## 技术栈

- **LLM**：DeepSeek（通过 OpenAI 兼容接口）
- **Embedding**：本地 `bge-small-zh-v1.5`（sentence-transformers，CPU 推理）
- **向量库**：Chroma（本地持久化）
- **文本切片**：自实现递归字符切片（参考 LangChain RecursiveCharacterTextSplitter 原理）
- **运行环境**：Python 3.10+，无框架依赖

## 快速开始

**1. 安装依赖**

```bash
cd practice
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r layer2/requirements.txt
```

**2. 配置 API Key**

在项目根目录（`practice/` 的上两级，即 `second-brain/`）创建 `.env` 文件：

```env
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_API_BASE=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

**3. 导入知识库**

```bash
cd layer3
python ingest.py
```

**4. 启动对话终端**

```bash
# 完整版（Layer 6）：Router + 语义缓存 + RAG + 记忆持久化
cd layer6
python chat_layer6.py

# 仅 RAG + 记忆持久化（Layer 5）
cd layer5
python chat_memory.py

# 仅 RAG（Layer 4）
cd layer4
python chat_rag.py

# 纯对话（Layer 1）
cd layer1
python chat_mine.py
```

## 各层核心设计

**Layer 3**：用文本内容的 SHA256 哈希作为文档 ID，配合 `upsert` 实现幂等导入；`ingest.py` 每次全量重建知识库，确保删除的内容不会残留。

**Layer 4**：RAG 流水线将检索片段通过 `role: system` 注入 prompt，位置在对话历史之后、当前问题之前，基于 Lost in the Middle 注意力分布原理。

**Layer 5**：对话历史以 JSON 文件按 session 持久化，发送给 API 的消息使用滑动窗口截断，存盘使用完整历史，两份数据独立维护。

**Layer 6**：Router 使用 LLM 做意图分类（`chitchat` / `rag` / `followup`），`max_tokens=8` 从 token 层面约束输出格式；语义缓存按意图分级阈值（闲聊 0.85，知识问答 0.90），`followup` 跳过缓存直接走对话历史。

## 已知局限与待优化项

- **混合意图**：Router 当前输出单一标签，"查一下知识库，上一条说的不对"这类同时包含 `followup` 和 `rag` 的输入无法正确处理。待优化为多标签输出。
- **缓存持久化**：语义缓存存在内存中，重启即清空。实际项目建议替换为 Redis（同时解决性能和持久化问题）。
- **跨语言语义**：bge-small-zh 为中文模型，"你好" vs "hello" 余弦相似度仅 0.471，跨语言语义缓存命中率极低。
- **知识库同步**：当前采用全量重建，实际项目建议改为差量同步（计算切片 ID 差集，只删除失效条目、新增变更条目）。

## 学习笔记

`data/` 目录下的两个文档是随项目同步更新的学习记录，同时也作为知识库的内容来源：

- `ai应用理论学习.md`：各层核心原理，解释"是什么、为什么"
- `实验结果总结.md`：各层实验数据、发现的 Bug 及设计决策
