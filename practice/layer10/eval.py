"""
Layer 10 - RAGAS 系统评测脚本

评测第十层 Agent + RAG 统一架构的整体质量。

使用 RAGAS 标准四维指标：
  - Faithfulness（忠实度）：回答是否忠实于检索到的上下文
  - Answer Relevancy（答案相关性）：回答与问题的相关度
  - Context Precision（上下文精确度）：检索内容中有多少是有用的
  - Context Recall（上下文召回率）：相比标准答案，检索内容覆盖了多少

工作流程：
  1. 加载评测数据集（eval_dataset.json）
  2. 逐条将 question 送入 Agent，收集 answer 和 contexts
  3. 组装 RAGAS 所需格式的数据
  4. 调用 RAGAS evaluate 计算各指标
  5. 输出结果报告

用法：
  cd practice/layer10
  python eval.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── 路径设置 ────────────────────────────────────────────────────
CURRENT_DIR  = Path(__file__).resolve().parent
PRACTICE_DIR = CURRENT_DIR.parent
LAYER1_DIR   = PRACTICE_DIR / "layer1"

# 确保 layer10 目录在 path 中
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_module(file_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 加载配置和组件 ──────────────────────────────────────────────
config = _load_module(LAYER1_DIR / "config_mine.py", "layer1_config")

from agent import run_agent  # noqa: E402
from tools import warmup  # noqa: E402


# ── 从 Agent 执行结果中提取 contexts ──────────────────────────
def extract_contexts(updated_msgs: list[dict]) -> list[str]:
    """
    从 Agent 执行后的消息列表中提取所有 search_knowledge_base 的返回内容。

    这些内容就是 RAGAS 评测中的 "contexts"。
    """
    contexts = []
    for msg in updated_msgs:
        if msg.get("role") != "tool":
            continue
        # 找到该 tool 消息对应的 tool_call_id
        tool_call_id = msg.get("tool_call_id", "")
        # 往前找对应的 assistant tool_calls 消息，确认工具名
        tool_name = _find_tool_name(updated_msgs, tool_call_id)
        if tool_name == "search_knowledge_base":
            content = msg.get("content", "").strip()
            if content and not content.startswith("[tool_error]"):
                contexts.append(content)
    return contexts


def _find_tool_name(messages: list[dict], tool_call_id: str) -> str:
    """根据 tool_call_id 从消息列表中找到对应的工具名。"""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc.get("id") == tool_call_id:
                return tc.get("function", {}).get("name", "")
    return ""


# ── 检测 Agent 调用了哪些工具 ─────────────────────────────────
def extract_tools_called(updated_msgs: list[dict]) -> list[str]:
    """从 Agent 消息中提取所有被调用的工具名。"""
    tools = []
    for msg in updated_msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tools.append(tc.get("function", {}).get("name", "?"))
    return tools


# ── 运行单条评测 ──────────────────────────────────────────────
def run_single_eval(question: str) -> dict:
    """
    将 question 送入 Agent，返回评测需要的数据。

    返回 dict 包含：
      - answer: Agent 的最终回答
      - contexts: 检索到的知识库内容列表
      - tools_called: 调用的工具名列表
      - latency: 响应耗时（秒）
    """
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    start_time = time.time()
    answer, updated_msgs = run_agent(messages)
    latency = time.time() - start_time

    contexts = extract_contexts(updated_msgs)
    tools_called = extract_tools_called(updated_msgs)

    return {
        "answer": answer,
        "contexts": contexts,
        "tools_called": tools_called,
        "latency": latency,
    }


# ── 构建 RAGAS 评判 LLM ───────────────────────────────────────
def _get_ragas_llm():
    """
    构建一个指向 DeepSeek API 的 ChatOpenAI 实例，供 RAGAS 作为评判模型使用。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.OPENAI_API_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_API_BASE,
        temperature=0,
        max_tokens=2048,
    )


# ── RAGAS 评测 ────────────────────────────────────────────────
def run_ragas_evaluation(results: list[dict]) -> dict:
    """
    使用 RAGAS 对收集到的结果做评测。

    只对有 contexts 的条目（即走了 RAG 的条目）计算三维指标：
      - Faithfulness（忠实度）
      - Context Precision（上下文精确度）
      - Context Recall（上下文召回率）

    注意：Answer Relevancy 因 DeepSeek API 不支持 n>1 参数而跳过。
    该指标内部需要一次生成多个变体问题（n=3），这与 DeepSeek 的限制冲突。

    返回各指标的均值。
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        return {"error": f"缺少依赖包: {exc}。请运行: pip install ragas datasets"}

    # 构建评判 LLM（指向 DeepSeek）
    judge_llm = _get_ragas_llm()

    # 只对有 contexts 的条目（即走了 RAG 的条目）做评测
    rag_items = [r for r in results if r.get("contexts")]

    report = {}

    if not rag_items:
        print("  [跳过] 没有 RAG 条目可评测")
        return report

    rag_data = {
        "question": [r["question"] for r in rag_items],
        "answer": [r["answer"] for r in rag_items],
        "contexts": [r["contexts"] for r in rag_items],
        "ground_truth": [r["ground_truth"] for r in rag_items],
    }
    rag_dataset = Dataset.from_dict(rag_data)

    metrics_list = [faithfulness, context_precision, context_recall]
    metric_names = ["faithfulness", "context_precision", "context_recall"]

    print(f"\n[RAGAS] 正在评估 {len(rag_items)} 条 RAG 请求的三维指标...")
    print(f"  评判模型: {config.OPENAI_API_MODEL} @ {config.OPENAI_API_BASE}")
    print(f"  指标: {', '.join(metric_names)}")
    print(f"  (跳过 Answer Relevancy: DeepSeek 不支持 n>1)")

    try:
        rag_result = evaluate(
            rag_dataset,
            metrics=metrics_list,
            llm=judge_llm,
            raise_exceptions=False,
        )

        # RAGAS 0.4.x 的 EvaluationResult:
        #   - __getitem__ 返回每条样本的分数列表，如 [0.8, 0.9, ...]
        #   - __repr__ 显示均值
        # 安全提取每个指标
        for name in metric_names:
            try:
                scores = rag_result[name]  # list of per-sample scores
                if isinstance(scores, (list, tuple)):
                    # 过滤掉 None/NaN
                    valid_vals = [v for v in scores if v is not None and v == v]
                    report[name] = sum(valid_vals) / len(valid_vals) if valid_vals else None
                    report[f"{name}_details"] = scores
                else:
                    report[name] = float(scores)
            except (KeyError, TypeError):
                report[name] = None

        report["num_samples"] = len(rag_items)

        for name in metric_names:
            score = report.get(name)
            if score is not None:
                print(f"  {name:25s}: {score:.4f}")
            else:
                print(f"  {name:25s}: N/A (计算失败)")

    except Exception as exc:
        report["error"] = str(exc)
        print(f"  [错误] RAGAS 指标计算失败: {exc}")

    return report


# ── 工具选择正确率（自定义指标）────────────────────────────────
def tool_selection_accuracy(results: list[dict]) -> dict:
    """
    评估 Agent 是否选择了正确的工具。

    对比 expected_tool（数据集标注）和 actual tools_called。
    """
    correct = 0
    total = len(results)
    details = []

    for r in results:
        expected = r.get("expected_tool")
        actual_tools = r.get("tools_called", [])

        if expected is None:
            # 期望不调用工具
            is_correct = len(actual_tools) == 0
        else:
            # 期望调用特定工具
            is_correct = expected in actual_tools

        if is_correct:
            correct += 1

        details.append({
            "question": r["question"][:40],
            "expected": expected or "(none)",
            "actual": actual_tools or ["(none)"],
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


# ── 主流程 ────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Second Brain —— 第十层 RAGAS 系统评测")
    print("=" * 60)

    # 1. 加载评测数据集
    dataset_path = CURRENT_DIR / "eval_dataset.json"
    if not dataset_path.exists():
        print(f"[错误] 评测数据集不存在: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n[数据集] 加载 {len(dataset)} 条测试样本")
    rag_count = sum(1 for d in dataset if d.get("expected_tool") == "search_knowledge_base")
    print(f"  - RAG 类: {rag_count} 条")
    print(f"  - 非 RAG 类: {len(dataset) - rag_count} 条")

    # 2. 预热
    print("\n[预热] 加载模型和索引...")
    warmup()
    print("[预热] 完成")

    # 3. 逐条运行 Agent 收集结果
    print(f"\n[评测] 开始逐条评测...")
    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_tool = item.get("expected_tool")

        print(f"\n  [{i}/{len(dataset)}] {question[:50]}...")

        eval_result = run_single_eval(question)
        eval_result["question"] = question
        eval_result["ground_truth"] = ground_truth
        eval_result["expected_tool"] = expected_tool

        tools_str = " → ".join(eval_result["tools_called"]) if eval_result["tools_called"] else "(无)"
        print(f"         工具: {tools_str}")
        print(f"         耗时: {eval_result['latency']:.1f}s")
        print(f"         回答: {eval_result['answer'][:60]}...")

        results.append(eval_result)

    # 4. 工具选择正确率
    print("\n" + "=" * 60)
    print("  评测结果")
    print("=" * 60)

    tool_report = tool_selection_accuracy(results)
    print(f"\n[工具选择正确率] {tool_report['accuracy']:.1%} ({tool_report['correct']}/{tool_report['total']})")
    for d in tool_report["details"]:
        status = "✓" if d["correct"] else "✗"
        print(f"  {status} {d['question']} | 期望={d['expected']} 实际={d['actual']}")

    # 5. 延迟统计
    latencies = [r["latency"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n[延迟统计]")
    print(f"  平均: {avg_latency:.1f}s")
    print(f"  最快: {min(latencies):.1f}s")
    print(f"  最慢: {max(latencies):.1f}s")

    # 6. RAGAS 评测
    print("\n[RAGAS 评测]")
    ragas_report = run_ragas_evaluation(results)

    # 7. 保存完整报告
    report = {
        "dataset_size": len(dataset),
        "tool_selection": tool_report,
        "latency": {
            "avg": avg_latency,
            "min": min(latencies),
            "max": max(latencies),
        },
        "ragas": ragas_report,
        "raw_results": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "ground_truth": r["ground_truth"],
                "expected_tool": r["expected_tool"],
                "tools_called": r["tools_called"],
                "contexts": r["contexts"],
                "latency": r["latency"],
            }
            for r in results
        ],
    }

    report_path = CURRENT_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] 完整评测报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("  评测完成")
    print("=" * 60)


def ragas_only() -> None:
    """从已有的 eval_report.json 中读取 raw_results，只重新运行 RAGAS 评测。"""
    print("=" * 60)
    print("  Second Brain —— RAGAS 重新评测（从已有报告）")
    print("=" * 60)

    report_path = CURRENT_DIR / "eval_report.json"
    if not report_path.exists():
        print(f"[错误] 报告文件不存在: {report_path}")
        print("  请先运行完整评测: python eval.py")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        old_report = json.load(f)

    results = old_report.get("raw_results", [])
    if not results:
        print("[错误] 报告中没有 raw_results")
        return

    print(f"\n[数据] 从报告中加载 {len(results)} 条结果")

    # 运行 RAGAS 评测
    print("\n[RAGAS 评测]")
    ragas_report = run_ragas_evaluation(results)

    # 更新报告的 ragas 部分
    old_report["ragas"] = ragas_report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(old_report, f, ensure_ascii=False, indent=2)
    print(f"\n[报告] 已更新: {report_path}")

    print("\n" + "=" * 60)
    print("  评测完成")
    print("=" * 60)


if __name__ == "__main__":
    if "--ragas-only" in sys.argv:
        ragas_only()
    else:
        main()
