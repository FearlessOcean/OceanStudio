"""自动化测试程序 - 批量评测 test_cases.json 中的40条语料"""
import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.agent import SmartHomeAgent
from src.knowledge_base import retrieve

console = Console()

TEST_CASES_PATH = Path(__file__).parent / "data" / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "data" / "test_results.json"


@dataclass
class CaseResult:
    id: str
    category: str
    difficulty: str
    description: str
    input: str
    expected_tools: list
    actual_tools: list
    tool_match: bool
    args_match: bool
    rag_match: bool
    response: str
    latency_ms: float
    passed: bool
    fail_reason: str = ""


def check_args(actual_tools: list[dict], expected_args: dict) -> bool:
    """检查实际工具调用参数是否包含所有期望的 key-value。"""
    if not expected_args:
        return True
    all_args = {}
    for tc in actual_tools:
        all_args.update(tc.get("args", {}))
    for k, v in expected_args.items():
        actual_val = all_args.get(k)
        if actual_val is None:
            return False
        # 字符串做包含匹配，数字做精确匹配
        if isinstance(v, str):
            if v not in str(actual_val):
                return False
        else:
            if actual_val != v:
                return False
    return True


def check_rag(response: str, rag_keywords: list[str]) -> bool:
    """检查回复中是否包含期望的 RAG 关键词。"""
    if not rag_keywords:
        return True
    return any(kw in response for kw in rag_keywords)


def run_test_case(agent: SmartHomeAgent, case: dict, session_agents: dict) -> CaseResult:
    """运行单条测试用例。多轮对话用 session_id 复用同一 agent 实例。"""
    import src.agent as agent_module

    session_id = case.get("session_id")
    if session_id:
        if session_id not in session_agents:
            session_agents[session_id] = SmartHomeAgent()
        test_agent = session_agents[session_id]
    else:
        # fresh_context=True：禁用 KV-cache，消除 Ollama 跨用例状态污染
        test_agent = SmartHomeAgent(fresh_context=True)

    # 拦截 _execute_tool_calls 记录实际调用
    actual_tools: list[dict] = []
    original_execute = test_agent._execute_tool_calls

    def _patched_execute(tool_calls):
        for tc in tool_calls:
            actual_tools.append({"name": tc["name"], "args": tc.get("args", {})})
        return original_execute(tool_calls)

    test_agent._execute_tool_calls = _patched_execute

    t0 = time.perf_counter()
    try:
        response = test_agent.chat(case["input"])
    except Exception as e:
        response = f"[ERROR] {e}"
    finally:
        test_agent._execute_tool_calls = original_execute  # 还原
    latency_ms = (time.perf_counter() - t0) * 1000

    actual_tool_names = [t["name"] for t in actual_tools]
    expected_tools = case.get("expected_tools", [])
    tool_optional = case.get("tool_optional", False)

    # 工具名匹配：期望的工具都被调用了（顺序无关）；tool_optional=True 时工具调用结果不影响判定
    if tool_optional:
        tool_match = True
    else:
        tool_match = set(expected_tools) == set(actual_tool_names) if expected_tools else (actual_tool_names == [])

    # 参数匹配
    args_match = check_args(actual_tools, case.get("expected_args_contain", {}))

    # RAG关键词匹配
    rag_keywords = case.get("expected_rag_keywords", [])
    rag_match = check_rag(response, rag_keywords)

    passed = tool_match and args_match and rag_match

    fail_reason = ""
    if not tool_match:
        fail_reason += f"工具不匹配(期望{expected_tools}，实际{actual_tool_names}) "
    if not args_match:
        fail_reason += f"参数不匹配 "
    if not rag_match:
        fail_reason += f"RAG关键词缺失{rag_keywords} "

    # 独立用例已用新实例，无需 reset

    return CaseResult(
        id=case["id"],
        category=case["category"],
        difficulty=case["difficulty"],
        description=case["description"],
        input=case["input"],
        expected_tools=expected_tools,
        actual_tools=actual_tool_names,
        tool_match=tool_match,
        args_match=args_match,
        rag_match=rag_match,
        response=response[:200],
        latency_ms=round(latency_ms, 1),
        passed=passed,
        fail_reason=fail_reason.strip(),
    )


def print_summary(results: list[CaseResult]):
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # 按类别统计
    by_category: dict[str, dict] = {}
    by_difficulty: dict[str, dict] = {}
    for r in results:
        for key, group in [(r.category, by_category), (r.difficulty, by_difficulty)]:
            if key not in group:
                group[key] = {"total": 0, "passed": 0}
            group[key]["total"] += 1
            if r.passed:
                group[key]["passed"] += 1

    # 总体结果
    color = "green" if passed / total >= 0.8 else "yellow" if passed / total >= 0.6 else "red"
    console.print(Panel.fit(
        f"[bold {color}]总通过率：{passed}/{total}  ({passed/total*100:.1f}%)[/bold {color}]",
        title="[bold]测试结果汇总[/bold]",
        border_style=color,
    ))

    # 按类别
    cat_table = Table(title="按类别统计", box=box.ROUNDED, border_style="cyan")
    cat_table.add_column("类别", style="cyan")
    cat_table.add_column("通过/总数", justify="center")
    cat_table.add_column("通过率", justify="center")
    for cat, stat in sorted(by_category.items()):
        rate = stat["passed"] / stat["total"]
        color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
        cat_table.add_row(cat, f"{stat['passed']}/{stat['total']}", f"[{color}]{rate*100:.0f}%[/{color}]")
    console.print(cat_table)

    # 按难度
    diff_table = Table(title="按难度统计", box=box.ROUNDED, border_style="magenta")
    diff_table.add_column("难度", style="magenta")
    diff_table.add_column("通过/总数", justify="center")
    diff_table.add_column("通过率", justify="center")
    for diff in ["easy", "medium", "hard"]:
        stat = by_difficulty.get(diff, {"total": 0, "passed": 0})
        if stat["total"] == 0:
            continue
        rate = stat["passed"] / stat["total"]
        color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
        diff_table.add_row(diff, f"{stat['passed']}/{stat['total']}", f"[{color}]{rate*100:.0f}%[/{color}]")
    console.print(diff_table)

    # 延迟统计
    latencies = [r.latency_ms for r in results]
    perf_table = Table(title="性能统计", box=box.ROUNDED, border_style="yellow")
    perf_table.add_column("指标", style="cyan")
    perf_table.add_column("值", style="yellow")
    perf_table.add_row("平均延迟", f"{sum(latencies)/len(latencies):.0f} ms")
    perf_table.add_row("最小延迟", f"{min(latencies):.0f} ms")
    perf_table.add_row("最大延迟", f"{max(latencies):.0f} ms")
    console.print(perf_table)

    # 失败用例详情
    failed_cases = [r for r in results if not r.passed]
    if failed_cases:
        fail_table = Table(title=f"失败用例详情（{len(failed_cases)}条）", box=box.ROUNDED, border_style="red")
        fail_table.add_column("ID", style="dim", width=5)
        fail_table.add_column("类别", width=8)
        fail_table.add_column("难度", width=6)
        fail_table.add_column("描述", width=20)
        fail_table.add_column("失败原因", style="red", no_wrap=False)
        for r in failed_cases:
            fail_table.add_row(r.id, r.category, r.difficulty, r.description, r.fail_reason)
        console.print(fail_table)


def save_results(results: list[CaseResult]):
    data = []
    for r in results:
        data.append({
            "id": r.id,
            "category": r.category,
            "difficulty": r.difficulty,
            "description": r.description,
            "input": r.input,
            "expected_tools": r.expected_tools,
            "actual_tools": r.actual_tools,
            "tool_match": r.tool_match,
            "args_match": r.args_match,
            "rag_match": r.rag_match,
            "passed": r.passed,
            "fail_reason": r.fail_reason,
            "latency_ms": r.latency_ms,
            "response": r.response,
        })
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[dim]结果已保存至 {RESULTS_PATH}[/dim]")


def run():
    console.print(Panel.fit(
        "[bold cyan]星护 Agent · 自动化测试程序[/bold cyan]\n"
        "[dim]批量评测 test_cases.json 中的40条语料[/dim]",
        border_style="cyan",
    ))

    test_cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    console.print(f"\n已加载 [bold]{len(test_cases)}[/bold] 条测试用例，开始测试...\n")

    agent = SmartHomeAgent()
    session_agents: dict[str, SmartHomeAgent] = {}
    results: list[CaseResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("测试进行中...", total=len(test_cases))

        for case in test_cases:
            progress.update(task, description=f"[{case['id']}] {case['description'][:20]}")
            result = run_test_case(agent, case, session_agents)
            results.append(result)
            status = "[green]OK[/green]" if result.passed else "[red]NG[/red]"
            console.print(f"  {status} {result.id} | {result.category} | {result.difficulty} | {result.latency_ms:.0f}ms | {result.description}")
            progress.advance(task)

    console.print()
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    run()
