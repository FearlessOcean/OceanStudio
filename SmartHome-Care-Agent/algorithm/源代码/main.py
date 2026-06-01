"""智能家居与健康看护 Agent - 入口程序"""
import sys
import os

# 强制 UTF-8 I/O，兼容 Windows GBK 终端和 Linux SSH 会话
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from src.agent import SmartHomeAgent
from src.monitor import MonitorDaemon
from src.knowledge_base import warmup_rag

console = Console()

# 预设演示场景（覆盖6个工具、多轮对话、RAG召回）
DEMO_SCENARIOS = [
    {
        "title": "复合任务：健康记录 + 设备控制",
        "input": "我奶奶今天有点发烧，帮我在系统里记录一下，另外马上把她卧室的空调调到26度。",
    },
    {
        "title": "异常告警：高烧紧急处置",
        "input": "奶奶体温量了39.8度，已经烧了3个小时了，怎么办？帮我记录并告警。",
    },
    {
        "title": "用药管理：设置提醒",
        "input": "帮爷爷设置一个用药提醒，他每天早晚要各吃一片降压药。",
    },
    {
        "title": "多轮对话：查询历史档案",
        "input": "查一下奶奶最近的健康记录。",
    },
    {
        "title": "环境感知 + 设备联动",
        "input": "帮我读一下客厅的室温，再读一下客厅的空气质量，然后告诉我结果。",
    },
    {
        "title": "知识问答（RAG召回）：血压异常处置",
        "input": "奶奶刚测血压180/100，帮我记录这个健康状态，并触发血压异常告警，告警详情写'收缩压180mmHg，舒张压100mmHg'。",
    },
]


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]星护 · 智能家居与健康看护 Agent[/bold cyan]\n"
        "[dim]端侧轻量化部署 | 模型: qwen2.5:1.5b | 内存: ~1.5GB[/dim]",
        border_style="cyan",
    ))


def run_demo(agent: SmartHomeAgent):
    """自动运行全部预设场景，展示 Agent 能力矩阵。"""
    console.print("\n[bold yellow]── 自动演示模式：运行 6 个预设场景 ──[/bold yellow]\n")

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        console.print(Panel(
            f"[bold]{scenario['input']}[/bold]",
            title=f"[cyan]场景 {i}：{scenario['title']}[/cyan]",
            border_style="blue",
        ))

        response = agent.chat(scenario["input"])

        console.print(Panel(
            f"[green]{response}[/green]",
            title="[green]Agent 回复[/green]",
            border_style="green",
        ))
        console.print()

    # 打印性能报告
    report = agent.profiler.report()
    table = Table(title="性能基准报告", box=box.ROUNDED, border_style="cyan")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="yellow")
    for k, v in report.items():
        table.add_row(k, str(v))
    console.print(table)


def run_interactive(agent: SmartHomeAgent):
    """交互式多轮对话模式。"""
    console.print("\n[bold yellow]── 交互模式（输入 'quit' 退出，'reset' 清除历史，'demo' 运行演示）──[/bold yellow]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]你[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]对话历史已清除。[/dim]")
            continue
        if user_input.lower() == "demo":
            run_demo(agent)
            continue

        response = agent.chat(user_input)
        console.print(Panel(f"[green]{response}[/green]", title="[green]星护[/green]", border_style="green"))


def _make_monitor_callback():
    """生成监控事件回调，将后台事件打印到控制台。"""
    EVENT_STYLES = {
        "SYSTEM":   ("bold blue",   "🔵"),
        "SCAN":     ("dim",         "🔍"),
        "OK":       ("dim green",   "✓"),
        "ALERT":    ("bold red",    "🚨"),
        "REMINDER": ("bold yellow", "💊"),
        "ERROR":    ("red",         "✗"),
    }

    def on_event(event_type: str, message: str):
        if event_type in ("SCAN", "OK", "SYSTEM"):
            return  # 日常巡检不打断交互，dashboard 可查看
        style, icon = EVENT_STYLES.get(event_type, ("white", "•"))
        console.print(f"[{style}]{icon} [Monitor] {message}[/{style}]")

    return on_event


def run():
    print_banner()
    rag_thread = warmup_rag()   # 后台并行预热嵌入模型
    agent = SmartHomeAgent()
    rag_thread.join(timeout=60) # 等待预热完成（最多60s）

    # 启动后台主动监控守护线程
    daemon = MonitorDaemon(on_event=_make_monitor_callback())
    daemon.start()

    # 判断是否在非交互环境（CI/管道）中运行
    if not sys.stdin.isatty():
        run_demo(agent)
    else:
        console.print("\n[1] 自动演示所有场景\n[2] 进入交互对话")
        choice = Prompt.ask("选择模式", choices=["1", "2"], default="1")
        if choice == "1":
            run_demo(agent)
        else:
            run_interactive(agent)

    console.print("\n[bold cyan]感谢使用星护 Agent！[/bold cyan]")


if __name__ == "__main__":
    run()
