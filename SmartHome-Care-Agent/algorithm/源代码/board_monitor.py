"""
星护 Agent · LuBanCat4 实时性能监控面板
在 Windows 上运行，通过 SSH 实时获取板子指标。

启动：
    pip install streamlit paramiko
    streamlit run algorithm/board_monitor.py
"""

import time
import threading
from collections import deque
from datetime import datetime

import paramiko
import streamlit as st

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="星护 · 板子性能监控",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 侧边栏：连接设置 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔌 连接设置")
    host     = st.text_input("板子 IP",   value="192.168.137.50")
    user     = st.text_input("用户名",    value="cat")
    password = st.text_input("密码",      value="ocean2001", type="password")
    interval = st.slider("刷新间隔（秒）", 1, 10, 2)
    st.divider()
    st.markdown("**硬件信息**")
    st.markdown("- 板子：LuBanCat4 (RK3588S)")
    st.markdown("- CPU：4×A76 + 4×A55")
    st.markdown("- NPU：6 TOPS")
    st.markdown("- 内存：4GB LPDDR4X")


# ── SSH 数据采集 ──────────────────────────────────────────────────────────────

COLLECT_CMD = r"""
echo '===CPU===' && \
  cat /proc/stat | head -9 && \
echo '===MEM===' && \
  free -m | grep Mem && \
echo '===FREQ_BIG===' && \
  for c in 4 5 6 7; do \
    cat /sys/devices/system/cpu/cpufreq/policy${c}/scaling_cur_freq 2>/dev/null || \
    cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_cur_freq 2>/dev/null || \
    echo 0; \
  done && \
echo '===FREQ_LITTLE===' && \
  for c in 0 1 2 3; do \
    cat /sys/devices/system/cpu/cpufreq/policy${c}/scaling_cur_freq 2>/dev/null || \
    cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_cur_freq 2>/dev/null || \
    echo 0; \
  done && \
echo '===TEMP===' && \
  for f in /sys/class/thermal/thermal_zone*/temp; do cat $f 2>/dev/null || echo 0; done && \
echo '===GPU===' && \
  cat /sys/class/devfreq/*/cur_freq 2>/dev/null | head -3 && \
echo '===NPU_FREQ===' && \
  (for p in /sys/class/devfreq/fdab0000.npu /sys/class/devfreq/*npu*; do \
    [ -f "$p/cur_freq" ] && cat "$p/cur_freq" 2>/dev/null && break; \
  done || echo 0) && \
echo '===NPU_LOAD===' && \
  (sudo cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo '') && \
echo '===PROC===' && \
  ps -eo pid,pcpu,pmem,comm --no-headers | grep -E 'python3?$' | head -5 && \
echo '===DONE==='
"""

_prev_cpu_stat: list[int] = []


def _parse_cpu_usage(stat_lines: list[str]) -> float:
    """从 /proc/stat 计算 CPU 总体使用率（差分）。"""
    global _prev_cpu_stat
    for line in stat_lines:
        if line.startswith("cpu "):
            vals = list(map(int, line.split()[1:]))
            if _prev_cpu_stat:
                prev, cur = _prev_cpu_stat, vals
                idle_delta  = (cur[3] - prev[3]) + (cur[4] - prev[4])
                total_delta = sum(cur[i] - prev[i] for i in range(min(len(cur), len(prev))))
                usage = 100.0 * (1 - idle_delta / max(total_delta, 1))
                _prev_cpu_stat = vals
                return round(usage, 1)
            _prev_cpu_stat = vals
            return 0.0
    return 0.0


def fetch_metrics(host: str, user: str, password: str) -> dict:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password, timeout=5)
        _, stdout, _ = client.exec_command(COLLECT_CMD, timeout=8)
        raw = stdout.read().decode(errors="replace")
        client.close()
    except Exception as e:
        return {"error": str(e)}

    # 解析各段
    sections: dict[str, list[str]] = {}
    cur_key = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("===") and line.endswith("==="):
            cur_key = line[3:-3]
            sections[cur_key] = []
        elif cur_key and line:
            sections[cur_key].append(line)

    metrics: dict = {"time": datetime.now().strftime("%H:%M:%S")}

    # CPU 使用率
    metrics["cpu_pct"] = _parse_cpu_usage(sections.get("CPU", []))

    # 内存
    mem_line = sections.get("MEM", [""])[0].split()
    if len(mem_line) >= 3:
        metrics["mem_total"] = int(mem_line[1])
        metrics["mem_used"]  = int(mem_line[2])
        metrics["mem_pct"]   = round(int(mem_line[2]) / max(int(mem_line[1]), 1) * 100, 1)
    else:
        metrics["mem_total"] = metrics["mem_used"] = metrics["mem_pct"] = 0

    # 大核频率（A76，核 4-7）
    big_freqs = []
    for v in sections.get("FREQ_BIG", []):
        try:
            big_freqs.append(round(int(v) / 1_000_000, 2))  # GHz
        except ValueError:
            big_freqs.append(0.0)
    metrics["big_freqs"] = big_freqs if big_freqs else [0.0] * 4

    # 小核频率（A55，核 0-3）
    little_freqs = []
    for v in sections.get("FREQ_LITTLE", []):
        try:
            little_freqs.append(round(int(v) / 1_000_000, 2))
        except ValueError:
            little_freqs.append(0.0)
    metrics["little_freqs"] = little_freqs if little_freqs else [0.0] * 4

    # 温度
    temps = []
    for v in sections.get("TEMP", []):
        try:
            temps.append(round(int(v) / 1000, 1))
        except ValueError:
            pass
    metrics["temps"] = temps

    # GPU 频率
    gpu_freqs = []
    for v in sections.get("GPU", []):
        try:
            gpu_freqs.append(round(int(v) / 1_000_000, 0))  # MHz
        except ValueError:
            pass
    metrics["gpu_freqs"] = gpu_freqs

    # NPU 频率
    npu_freq_raw = sections.get("NPU_FREQ", ["0"])
    try:
        metrics["npu_freq_mhz"] = int(int(npu_freq_raw[0]) / 1_000_000)
    except (ValueError, IndexError):
        metrics["npu_freq_mhz"] = 0

    # NPU 负载（Core0/1/2 各核占用率）
    import re as _re
    npu_loads = [0, 0, 0]
    for line in sections.get("NPU_LOAD", []):
        matches = _re.findall(r"Core\d:\s*(\d+)%", line)
        if matches:
            npu_loads = [int(x) for x in matches[:3]]
            while len(npu_loads) < 3:
                npu_loads.append(0)
            break
    metrics["npu_loads"]    = npu_loads
    metrics["npu_avg_load"] = sum(npu_loads) // max(len(npu_loads), 1)

    # 进程
    metrics["procs"] = sections.get("PROC", [])

    return metrics


# ── 历史数据 ──────────────────────────────────────────────────────────────────
MAX_HIST = 60  # 最多保留 60 个数据点
hist_time      = deque(maxlen=MAX_HIST)
hist_cpu       = deque(maxlen=MAX_HIST)
hist_mem       = deque(maxlen=MAX_HIST)
hist_big_freq  = deque(maxlen=MAX_HIST)
hist_npu_load  = deque(maxlen=MAX_HIST)


# ── 主界面 ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4A90D9;'>
    🌟 星护 Agent · LuBanCat4 实时性能监控
</h1>
<p style='text-align:center; color:#888;'>RK3588S · 4GB LPDDR4X · 6 TOPS NPU · Ubuntu 22.04</p>
""", unsafe_allow_html=True)

status_bar  = st.empty()
main_area   = st.empty()

# ── 主循环 ────────────────────────────────────────────────────────────────────
while True:
    m = fetch_metrics(host, user, password)

    if "error" in m:
        status_bar.error(f"❌ 连接失败：{m['error']}")
        time.sleep(interval)
        continue

    status_bar.success(f"✅ 已连接  |  更新时间：{m['time']}")

    # 追加历史
    hist_time.append(m["time"])
    hist_cpu.append(m["cpu_pct"])
    hist_mem.append(m["mem_pct"])
    avg_big = sum(m["big_freqs"]) / max(len(m["big_freqs"]), 1)
    hist_big_freq.append(avg_big)
    hist_npu_load.append(m["npu_avg_load"])

    with main_area.container():

        # ── 第一行：核心指标卡片（5列）────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric("🖥️ CPU 使用率", f"{m['cpu_pct']} %",
                      delta=f"{m['cpu_pct'] - (hist_cpu[-2] if len(hist_cpu) > 1 else m['cpu_pct']):.1f}%")
            st.progress(min(m["cpu_pct"] / 100, 1.0))

        with c2:
            used_gb = m["mem_used"] / 1024
            total_gb = m["mem_total"] / 1024
            mem_delta = m["mem_pct"] - (hist_mem[-2] if len(hist_mem) > 1 else m["mem_pct"])
            st.metric("🧠 内存使用", f"{used_gb:.1f} / {total_gb:.1f} GB",
                      delta=f"{mem_delta:+.1f}%")
            st.progress(min(m["mem_pct"] / 100, 1.0))

        with c3:
            avg_big_ghz = sum(m["big_freqs"]) / max(len(m["big_freqs"]), 1)
            st.metric("⚡ A76 大核频率", f"{avg_big_ghz:.2f} GHz",
                      delta="4 核")
            st.progress(min(avg_big_ghz / 2.4, 1.0))

        with c4:
            max_temp = max(m["temps"]) if m["temps"] else 0
            temp_color = "🔴" if max_temp > 70 else "🟡" if max_temp > 55 else "🟢"
            st.metric(f"{temp_color} 最高温度", f"{max_temp} °C")
            st.progress(min(max_temp / 90, 1.0))

        with c5:
            npu_color = "🔴" if m["npu_avg_load"] > 80 else "🟡" if m["npu_avg_load"] > 30 else "🟢"
            npu_freq_label = f"{m['npu_freq_mhz']} MHz" if m["npu_freq_mhz"] > 0 else "N/A"
            st.metric(f"{npu_color} NPU 负载", f"{m['npu_avg_load']} %",
                      delta=npu_freq_label)
            st.progress(min(m["npu_avg_load"] / 100, 1.0))

        st.divider()

        # ── 第二行：时序图 ────────────────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("📈 CPU & 内存使用率趋势")
            if len(hist_time) > 1:
                import pandas as pd
                df = pd.DataFrame({
                    "时间":     list(hist_time),
                    "CPU (%)":  list(hist_cpu),
                    "内存 (%)": list(hist_mem),
                })
                st.line_chart(df.set_index("时间"))
            else:
                st.info("收集数据中...")

        with col_r:
            st.subheader("🤖 NPU 负载趋势 (%)")
            if len(hist_time) > 1:
                import pandas as pd
                df2 = pd.DataFrame({
                    "时间":       list(hist_time),
                    "NPU 平均负载 (%)": list(hist_npu_load),
                })
                st.line_chart(df2.set_index("时间"))
            else:
                st.info("收集数据中...")

        st.divider()

        # ── 第三行：详细指标 ─────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.subheader("🔧 各核频率 (GHz)")
            core_data = {}
            for i, f in enumerate(m["big_freqs"]):
                core_data[f"A76 核{i+4}"] = f
            for i, f in enumerate(m["little_freqs"]):
                core_data[f"A55 核{i}"] = f
            if core_data:
                import pandas as pd
                df3 = pd.DataFrame({"频率 (GHz)": core_data})
                st.bar_chart(df3)

        with col_b:
            st.subheader("🌡️ 温度分布 (°C)")
            if m["temps"]:
                import pandas as pd
                temp_dict = {f"Zone {i}": t for i, t in enumerate(m["temps"][:8])}
                df4 = pd.DataFrame({"温度 (°C)": temp_dict})
                st.bar_chart(df4)
            else:
                st.info("无温度数据")

        with col_c:
            st.subheader("🤖 NPU 各核负载")
            if any(m["npu_loads"]):
                import pandas as pd
                npu_df = pd.DataFrame({
                    "负载 (%)": {f"Core {i}": v for i, v in enumerate(m["npu_loads"])}
                })
                st.bar_chart(npu_df)
                freq_str = f"{m['npu_freq_mhz']} MHz" if m["npu_freq_mhz"] > 0 else "无法读取"
                st.caption(f"NPU 频率：{freq_str}  |  3 核心并行")
            else:
                st.info("NPU 空闲（负载读取需 debugfs 权限）")
                freq_str = f"{m['npu_freq_mhz']} MHz" if m["npu_freq_mhz"] > 0 else "N/A"
                st.caption(f"NPU 频率：{freq_str}")

            st.subheader("⚙️ 进程状态")
            if m["procs"]:
                for p in m["procs"]:
                    parts = p.split()
                    if len(parts) >= 4:
                        pid, cpu, mem, name = parts[0], parts[1], parts[2], parts[3]
                        st.markdown(
                            f"`{name[:20]:<20}` CPU:{cpu:>5}% MEM:{mem:>5}%"
                        )
            else:
                st.info("未检测到 Python 进程")

            if m["gpu_freqs"]:
                st.caption(f"Mali GPU：{m['gpu_freqs'][0]:.0f} MHz")

        st.divider()
        st.caption(f"📡 SSH → {host}  |  刷新间隔 {interval}s  |  数据点 {len(hist_time)}/{MAX_HIST}")

    time.sleep(interval)
