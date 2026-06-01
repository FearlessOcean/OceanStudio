"""主动监控守护进程 - 后台定时巡检传感器 + 用药提醒调度"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from .tools import read_environment_sensor, trigger_health_alert

# ── 阈值配置 ──────────────────────────────────────────────────────────────────

SENSOR_RULES: list[dict] = [
    {
        "room": "bedroom",
        "sensor_type": "temperature",
        "check": lambda v: _parse_float(v) > 35.0,
        "alert_type": "high_temperature",
        "member_name": "家庭成员",
        "detail_fn": lambda v: f"卧室温度过高：{v}，请注意防暑",
    },
    {
        "room": "living_room",
        "sensor_type": "temperature",
        "check": lambda v: _parse_float(v) > 35.0,
        "alert_type": "high_temperature",
        "member_name": "家庭成员",
        "detail_fn": lambda v: f"客厅温度过高：{v}，请注意防暑",
    },
    {
        "room": "living_room",
        "sensor_type": "air_quality",
        "check": lambda v: _parse_pm25(v) > 75,
        "alert_type": "poor_air_quality",
        "member_name": "家庭成员",
        "detail_fn": lambda v: f"客厅空气质量差：{v}，建议关窗开净化器",
    },
    {
        "room": "bedroom",
        "sensor_type": "air_quality",
        "check": lambda v: _parse_pm25(v) > 75,
        "alert_type": "poor_air_quality",
        "member_name": "家庭成员",
        "detail_fn": lambda v: f"卧室空气质量差：{v}，建议关窗开净化器",
    },
    {
        "room": "bathroom",
        "sensor_type": "humidity",
        "check": lambda v: _parse_float(v) > 85.0,
        "alert_type": "high_humidity",
        "member_name": "家庭成员",
        "detail_fn": lambda v: f"浴室湿度过高：{v}，跌倒风险上升，请注意安全",
    },
]

# 巡检间隔（秒）和用药提醒检查间隔（秒）
SENSOR_INTERVAL = 30
REMINDER_INTERVAL = 60

_DATA_DIR = Path(__file__).parent.parent / "data"
_REMINDER_DB = _DATA_DIR / "reminders.json"


# ── 辅助解析 ──────────────────────────────────────────────────────────────────

def _parse_float(value: str) -> float:
    """从传感器返回字符串中提取第一个浮点数。"""
    import re
    m = re.search(r"[\d.]+", str(value))
    return float(m.group()) if m else 0.0


def _parse_pm25(value: str) -> float:
    """从 PM2.5 字符串中提取数值，如 'PM2.5: 22 μg/m³（良）' → 22.0"""
    import re
    m = re.search(r"PM2\.5[:\s]*([\d.]+)", str(value))
    return float(m.group(1)) if m else 0.0


# ── 核心守护类 ────────────────────────────────────────────────────────────────

class MonitorDaemon:
    """后台监控守护进程，包含两个独立线程：
    1. sensor_loop  — 定时读传感器，超阈值触发告警
    2. reminder_loop — 扫描 reminders.json，到点输出用药提醒
    """

    def __init__(self, on_event=None):
        """
        Args:
            on_event: 可选回调 fn(event_type: str, message: str)，
                      供 main.py 将事件打印到控制台。
                      若为 None，事件只写到标准输出。
        """
        self._on_event = on_event or (lambda etype, msg: print(f"[Monitor][{etype}] {msg}"))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def start(self):
        """启动所有后台线程（daemon=True，主进程退出时自动停止）。"""
        for target, name in [
            (self._sensor_loop, "sensor-monitor"),
            (self._reminder_loop, "reminder-scheduler"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self._on_event("SYSTEM", f"主动监控已启动（传感器巡检每{SENSOR_INTERVAL}s，用药提醒每{REMINDER_INTERVAL}s）")

    def stop(self):
        """优雅停止所有后台线程。"""
        self._stop.set()

    # ── 传感器巡检循环 ────────────────────────────────────────────────────────

    def _sensor_loop(self):
        while not self._stop.is_set():
            self._check_all_sensors()
            self._stop.wait(SENSOR_INTERVAL)

    def _check_all_sensors(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._on_event("SCAN", f"[{timestamp}] 开始传感器巡检...")

        for rule in SENSOR_RULES:
            try:
                raw = read_environment_sensor.invoke({
                    "room": rule["room"],
                    "sensor_type": rule["sensor_type"],
                })
                # 提取读数部分（工具返回格式："bedroom 的 temperature 读数：23.5°C"）
                value = raw.split("读数：")[-1] if "读数：" in raw else raw

                if rule["check"](value):
                    detail = rule["detail_fn"](value)
                    alert_raw = trigger_health_alert.invoke({
                        "member_name": rule["member_name"],
                        "alert_type": rule["alert_type"],
                        "detail": detail,
                    })
                    self._on_event("ALERT", f"⚠ {detail}")
                else:
                    self._on_event(
                        "OK",
                        f"{rule['room']}/{rule['sensor_type']} = {value} （正常）",
                    )
            except Exception as e:
                self._on_event("ERROR", f"传感器读取失败 {rule['room']}/{rule['sensor_type']}: {e}")

    # ── 用药提醒调度循环 ──────────────────────────────────────────────────────

    def _reminder_loop(self):
        while not self._stop.is_set():
            self._check_reminders()
            self._stop.wait(REMINDER_INTERVAL)

    def _check_reminders(self):
        if not _REMINDER_DB.exists():
            return
        try:
            db: dict = json.loads(_REMINDER_DB.read_text(encoding="utf-8"))
        except Exception:
            return

        now = datetime.now()
        hour = now.hour

        for member, reminders in db.items():
            for reminder in reminders:
                if not reminder.get("active", True):
                    continue
                schedule: str = reminder.get("schedule", "")
                medication: str = reminder.get("medication", "")

                if _should_remind(schedule, hour):
                    self._on_event(
                        "REMINDER",
                        f"💊 {member} 用药提醒：{medication}（{schedule}）",
                    )


def _should_remind(schedule: str, current_hour: int) -> bool:
    """根据 schedule 文本和当前小时判断是否需要提醒。
    支持格式：
    - "每天早晚各一次" → 早8点、晚20点
    - "每天8:00和20:00"
    - "每天早上8点"
    - "每天三次" → 8、13、18点
    """
    import re

    # 提取所有明确的小时数，匹配 "8:00"、"8点"、"8时" 等格式
    hours_in_schedule = [int(h) for h in re.findall(r"(?<!\d)(\d{1,2})(?::00|点|时)", schedule)]

    if hours_in_schedule:
        return current_hour in hours_in_schedule

    # 关键词推断
    triggers = []
    if any(kw in schedule for kw in ["早", "早上", "早晨", "morning"]):
        triggers.append(8)
    if any(kw in schedule for kw in ["中午", "午"]):
        triggers.append(12)
    if any(kw in schedule for kw in ["晚", "晚上", "evening", "night"]):
        triggers.append(20)
    if "三次" in schedule and not triggers:
        triggers = [8, 13, 18]
    if "两次" in schedule and not triggers:
        triggers = [8, 20]

    return current_hour in triggers
