"""星护 Dashboard - Flask 后端，为前端提供实时状态 API"""
import json
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

_DATA_DIR = Path(__file__).parent / "data"

# 设备中文名映射
_DEVICE_NAMES = {
    "living_room_light": "客厅灯",
    "bedroom_light": "卧室灯",
    "bedroom_ac": "卧室空调",
    "living_room_ac": "客厅空调",
    "front_door_lock": "前门门锁",
    "curtain": "窗帘",
}

# 告警类型中文名
_ALERT_TYPES = {
    "fall_detected": "跌倒检测",
    "high_fever": "高烧告警",
    "abnormal_bp": "血压异常",
    "no_activity": "长时间无活动",
    "irregular_heartbeat": "心率异常",
}


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _read_alerts(n: int = 10) -> list:
    log_path = _DATA_DIR / "alerts.log"
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    return lines[-n:][::-1]  # 最新的在前


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    device_raw = _load_json(_DATA_DIR / "device_state.json")
    health_raw = _load_json(_DATA_DIR / "health_records.json")
    reminder_raw = _load_json(_DATA_DIR / "reminders.json")

    # 整理设备状态
    devices = []
    for key, info in device_raw.items():
        if key == "sensors":
            continue
        name = _DEVICE_NAMES.get(key, key)
        is_on = info.get("on")
        is_locked = info.get("locked")
        status_text = "未知"
        status_type = "unknown"
        if is_locked is not None:
            status_text = "已锁定" if is_locked else "已解锁"
            status_type = "locked" if is_locked else "unlocked"
        elif is_on is not None:
            status_text = "开启" if is_on else "关闭"
            status_type = "on" if is_on else "off"
        extra = ""
        if info.get("temperature"):
            extra = f" {info['temperature']}°C"
        elif info.get("brightness"):
            extra = f" 亮度{info['brightness']}%"
        devices.append({
            "id": key,
            "name": name,
            "status": status_text + extra,
            "type": status_type,
            "updated_at": info.get("updated_at", ""),
        })

    # 传感器数据
    sensors = []
    for key, info in device_raw.get("sensors", {}).items():
        sensors.append({
            "room": info.get("room", ""),
            "type": info.get("type", ""),
            "value": info.get("value", ""),
            "updated_at": info.get("updated_at", ""),
        })

    # 健康记录：每个成员最近3条
    health = []
    for member, records in health_raw.items():
        recent = records[-3:][::-1]
        health.append({
            "member": member,
            "records": [{"time": r["time"], "symptom": r["symptom"]} for r in recent],
        })

    # 用药提醒：只显示 active 的，同一成员+药物只保留最新一条
    reminders_seen = {}
    for member, items in reminder_raw.items():
        for r in items:
            if r.get("active", True):
                key = (member, r["medication"])
                reminders_seen[key] = {"member": member, "medication": r["medication"], "schedule": r["schedule"]}
    reminders = list(reminders_seen.values())

    # 告警日志
    alerts = _read_alerts(8)

    return jsonify({
        "devices": devices,
        "sensors": sensors,
        "health": health,
        "reminders": reminders,
        "alerts": alerts,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
