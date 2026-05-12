import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on local environment
    def load_dotenv(*args, **kwargs):
        return False

# 优先加载项目根目录 .env，也兼容本层目录单独放置 .env
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
load_dotenv(dotenv_path=CURRENT_DIR / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
OPENAI_API_MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
# Backward compatibility for previous typo in variable name.
EMBENDDING_MODEL = EMBEDDING_MODEL
SYSTEM_PROMPT = """你是一个严谨的知识库助手。
回答时简洁直接，避免寒暄。
对于知识库中不存在的知识，不要随意编造。"""
MAX_HISTORY_TURNS = 1
TEMPERATURE = 0.7
MAX_TOKENS = 2048
