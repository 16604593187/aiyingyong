# =============================================================================
# config.py —— 全局配置
# 职责：集中管理所有配置项，避免 API Key 等敏感信息散落在业务代码里
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env（若存在）
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ── 模型配置 ──────────────────────────────────────────────────────────────────

# API Key：优先从环境变量读取，其次直接填写（不推荐提交到 git）
# 推荐做法：export OPENAI_API_KEY="sk-xxx" 或在 .env 文件里设置
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# API Base URL：
#   - OpenAI 官方：https://api.openai.com/v1
#   - DeepSeek：   https://api.deepseek.com/v1
#   - 阿里云 Qwen：https://dashscope.aliyuncs.com/compatible-mode/v1
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")

# 使用的模型名称
#   - DeepSeek：deepseek-chat
#   - OpenAI：  gpt-4o-mini
#   - Qwen：    qwen-turbo
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# ── 对话配置 ──────────────────────────────────────────────────────────────────

# System Prompt：定义助手的角色和行为
SYSTEM_PROMPT = """你是一个专注、简洁的 AI 助手。
回答问题时直接切入重点，避免不必要的寒暄。
如果不确定答案，直接说不知道，不要编造内容。"""

# 对话历史最大保留轮数（超出后自动截断最早的对话）
# 1 轮 = 1 条 user 消息 + 1 条 assistant 消息
MAX_HISTORY_TURNS = 10

# 模型生成参数
TEMPERATURE = 0.7       # 创造性：0.0 最保守，1.0 最发散
MAX_TOKENS = 2048       # 单次回复最大 token 数
