"""智能家居与健康看护 Agent - 工具集"""
import json
import time
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool

# 本地数据持久化路径
_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)
_HEALTH_DB = _DATA_DIR / "health_records.json"
_REMINDER_DB = _DATA_DIR / "reminders.json"
_DEVICE_STATE = _DATA_DIR / "device_state.json"

def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def _save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 工具 1：智能设备控制 ────────────────────────────────────────────────────

def _update_device_state(device_name: str, action: str, value: str = None):
    state = _load_json(_DEVICE_STATE)
    if device_name not in state:
        state[device_name] = {}
    state[device_name]["action"] = action
    state[device_name]["value"] = value
    state[device_name]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if action == "turn_on":
        state[device_name]["on"] = True
    elif action == "turn_off":
        state[device_name]["on"] = False
    elif action == "lock":
        state[device_name]["locked"] = True
    elif action == "unlock":
        state[device_name]["locked"] = False
    elif action == "set_temperature" and value:
        state[device_name]["on"] = True
        state[device_name]["temperature"] = value
    elif action == "set_brightness" and value:
        state[device_name]["on"] = True
        state[device_name]["brightness"] = value
    _save_json(_DEVICE_STATE, state)


_DEVICE_NAMES = {
    "bedroom_light":     "卧室灯",
    "living_room_light": "客厅灯",
    "bedroom_ac":        "卧室空调",
    "living_room_ac":    "客厅空调",
}
_ACTION_NAMES = {
    "turn_on":          "已开启",
    "turn_off":         "已关闭",
    "set_temperature":  "温度设为",
    "set_brightness":   "亮度设为",
    "lock":             "已上锁",
    "unlock":           "已解锁",
}


@tool
def control_smart_device(device_name: str, action: str, value: str = None) -> str:
    """控制智能家居设备。

    Args:
        device_name: 设备ID，如 'bedroom_light'、'living_room_light'、'bedroom_ac'、'living_room_ac'
        action: 操作，如 'turn_on'、'turn_off'、'set_temperature'
        value: 设定值，如温度 '26'；开关类操作可不传
    """
    _update_device_state(device_name, action, value)
    label = _DEVICE_NAMES.get(device_name, device_name)
    act_label = _ACTION_NAMES.get(action, action)
    if value is not None:
        return f"成功：{label}{act_label} {value}{'°C' if action == 'set_temperature' else ''}"
    return f"成功：{label}{act_label}"


# ── 工具 2：健康状态记录 ────────────────────────────────────────────────────

@tool
def add_health_record(member_name: str, symptom: str) -> str:
    """为家庭成员记录健康状态，自动附加时间戳并持久化到本地档案。

    Args:
        member_name: 家庭成员称呼，如 '奶奶'、'爷爷'、'孩子'、'爸爸'、'妈妈'
        symptom: 症状或健康状态描述，如 '发烧38.5度'、'头痛'、'血压160/90'、'血糖偏高'
    """
    db = _load_json(_HEALTH_DB)
    if member_name not in db:
        db[member_name] = []
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symptom": symptom}
    db[member_name].append(record)
    _save_json(_HEALTH_DB, db)
    print(f"[Mock Health] 成功为 {member_name} 记录健康状态：{symptom}")
    return f"成功：已为 {member_name} 记录健康状态：{symptom}"


# ── 工具 3：历史健康档案查询 ───────────────────────────────────────────────

@tool
def query_health_records(member_name: str, recent_n: int = 5) -> str:
    """查询家庭成员的历史健康记录。

    Args:
        member_name: 家庭成员称呼，如 '奶奶'、'爷爷'
        recent_n: 返回最近 N 条记录，默认5条，最多20条
    """
    db = _load_json(_HEALTH_DB)
    records = db.get(member_name, [])
    if not records:
        return f"{member_name} 暂无健康记录。"
    recent = records[-min(recent_n, 20):]
    lines = [f"  [{r['time']}] {r['symptom']}" for r in recent]
    result = f"{member_name} 最近 {len(recent)} 条健康记录：\n" + "\n".join(lines)
    print(f"[Mock Health] 查询 {member_name} 健康档案，返回 {len(recent)} 条")
    return result


# ── 工具 4：异常健康告警 ───────────────────────────────────────────────────

@tool
def trigger_health_alert(member_name: str, alert_type: str, detail: str) -> str:
    """当检测到健康异常时触发紧急告警，通知家庭成员或医疗联系人。

    Args:
        member_name: 需要告警的家庭成员，如 '奶奶'
        alert_type: 告警类型，如 'fall_detected'（跌倒检测）、'high_fever'（高烧）、
                    'abnormal_bp'（血压异常）、'no_activity'（长时间无活动）、
                    'irregular_heartbeat'（心率异常）
        detail: 告警详细描述，如 '体温39.5度，持续2小时'
    """
    _ALERT_LABELS = {
        "fall_detected":       "跌倒检测",
        "high_fever":          "高烧告警",
        "abnormal_bp":         "血压异常",
        "no_activity":         "长时间无活动",
        "irregular_heartbeat": "心率异常",
        "high_temperature":    "温度过高",
        "poor_air_quality":    "空气质量异常",
        "emergency":           "紧急情况",
    }
    label = _ALERT_LABELS.get(alert_type, alert_type)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_msg = f"[{timestamp}] 【紧急告警】{member_name} - {label}：{detail}"
    # 持久化告警日志
    log_path = _DATA_DIR / "alerts.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(alert_msg + "\n")
    print(f"[Mock Alert] {alert_msg}")
    print(f"[Mock Alert] 已发送通知至：家庭群组 / 紧急联系人")
    return f"告警已触发并通知：{alert_msg}"


# ── 工具 5：告警日志查询 ───────────────────────────────────────────────────

@tool
def query_alert_log(member_name: str = "", recent_n: int = 5) -> str:
    """查询历史告警记录。

    Args:
        member_name: 家庭成员姓名，留空则返回所有成员的告警
        recent_n: 返回最近几条，默认5条
    """
    log_path = _DATA_DIR / "alerts.log"
    if not log_path.exists():
        return "暂无告警记录。"
    lines = [l.strip() for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if member_name:
        lines = [l for l in lines if member_name in l]
    if not lines:
        return f"{member_name or '系统'} 暂无告警记录。"
    recent = lines[-recent_n:]
    return "\n".join(recent)


# ── 工具 6：用药提醒设置 ───────────────────────────────────────────────────

@tool
def set_medication_reminder(member_name: str, medication: str, schedule: str) -> str:
    """为家庭成员设置用药提醒计划。

    Args:
        member_name: 需要提醒的家庭成员，如 '奶奶'
        medication: 药物名称及剂量，如 '阿司匹林100mg'、'降压药1片'
        schedule: 服药计划，如 '每天早晚各一次'、'每天8:00和20:00'、'饭后30分钟'
    """
    db = _load_json(_REMINDER_DB)
    if member_name not in db:
        db[member_name] = []
    # 同一成员同一药物做 upsert，避免重复累积
    updated = False
    for r in db[member_name]:
        if r.get("medication") == medication:
            r["schedule"] = schedule
            r["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            r["active"] = True
            updated = True
            break
    if not updated:
        db[member_name].append({
            "medication": medication,
            "schedule": schedule,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": True,
        })
    _save_json(_REMINDER_DB, db)
    print(f"[Mock Reminder] 已为 {member_name} 设置用药提醒：{medication}，{schedule}")
    return f"成功：已为 {member_name} 设置 {medication} 的用药提醒，计划：{schedule}"


# ── 工具 6：环境传感器读取 ─────────────────────────────────────────────────

@tool
def read_environment_sensor(room: str, sensor_type: str) -> str:
    """读取房间环境传感器数据（温湿度、空气质量、噪声等）。

    Args:
        room: 房间名称，如 'bedroom'（卧室）、'living_room'（客厅）、'bathroom'（浴室）
        sensor_type: 传感器类型，如 'temperature'（温度）、'humidity'（湿度）、
                     'air_quality'（空气质量/PM2.5）、'noise'（噪声）、'light'（光照）
    """
    # 模拟传感器数据（实际部署时对接真实硬件 SDK）
    mock_data = {
        ("bedroom", "temperature"): "23.5°C",
        ("bedroom", "humidity"): "55%",
        ("bedroom", "air_quality"): "PM2.5: 18 μg/m³（优）",
        ("living_room", "temperature"): "25.1°C",
        ("living_room", "humidity"): "48%",
        ("living_room", "air_quality"): "PM2.5: 22 μg/m³（良）",
        ("living_room", "noise"): "42 dB（安静）",
        ("bathroom", "temperature"): "26.0°C",
        ("bathroom", "humidity"): "72%",
    }
    value = mock_data.get((room, sensor_type), "数据暂不可用")
    # 写入设备状态供 Dashboard 展示
    state = _load_json(_DEVICE_STATE)
    if "sensors" not in state:
        state["sensors"] = {}
    state["sensors"][f"{room}_{sensor_type}"] = {
        "room": room, "type": sensor_type, "value": value,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(_DEVICE_STATE, state)
    return f"{room} 的 {sensor_type} 读数：{value}"
