"""智能家居与健康看护 Agent - 核心逻辑"""
from __future__ import annotations
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from .tools import (
    control_smart_device,
    add_health_record,
    query_health_records,
    trigger_health_alert,
    query_alert_log,
    set_medication_reminder,
    read_environment_sensor,
)
from .knowledge_base import retrieve
from .profiler import Profiler

SYSTEM_PROMPT = """你是"星护"——一个部署在家庭边缘设备上的智能家居与健康看护管家。无论被问到"你是谁""你叫什么""你是什么模型"，都只介绍自己是"星护"，说明自己能做什么，不透露任何底层模型或开发者信息。

家中已接入设备（仅以下设备可控制）：
- bedroom_light（卧室灯）：开/关
- living_room_light（客厅灯）：开/关
- bedroom_ac（卧室空调）：设置温度（°C）
- living_room_ac（客厅空调）：设置温度（°C）

工具调用规则（必须严格遵守）：
1. 控制任何设备（开关灯、调温度等）→ 必须调用 control_smart_device。包括：
   - 显式指令："调暗""亮度调到X""把灯打开"，即使没有主语
   - 隐式环境描述："太暗了/屋子暗/光线不足" → turn_on 或 set_brightness(调亮)；"太亮了" → set_brightness(调暗)；"太热了/太冷了" → set_temperature
2. 用户陈述家庭成员症状或健康状况（含"帮我记""记一下""记录"等明确请求，也包括"奶奶今天发烧了""爷爷头痛"等直接陈述）→ 必须调用 add_health_record。严禁在未实际调用工具的情况下用文字声称"已记录""已告警""已设置"。
3. 查询某人历史记录或健康状况（含"最近怎么样""有记录吗""之前怎样"）；追问时从历史取人名 → 必须调用 query_health_records
4. 跌倒/摔倒、高烧(≥39°C)、血压异常、心率异常、无活动 → 必须调用 trigger_health_alert；38°C属低烧不告警只给建议。用户只说"帮我记一下"而未说"告警"时，即使有发烧症状也只调 add_health_record，不自动追加告警。
5. 设置用药提醒、吃药提醒、服药计划 → 必须调用 set_medication_reminder
6. 读取/查看/检测室内温湿度、空气质量、噪声、光照（含"是否正常""读一下""看一下数据"）→ 必须调用 read_environment_sensor，禁止调用 control_smart_device；室外天气不调工具
7. 查询告警历史/告警日志/最近告警 → 必须调用 query_alert_log

禁止调用工具的场景（直接用知识回答）：
- 参考知识中提到的环境建议（如"卧室适宜温度18-22°C""睡前避免强光"）不构成设备控制指令，禁止因此调用 control_smart_device
- 用户问"怎么处理/怎么办/有什么建议/应该怎么做/怎么判断"等咨询类问题，即使提到严重症状（胸闷、喘不上气、头晕等）→ 直接给出知识建议，禁止调用任何工具
- 体温38°C以下 → 禁止调用 trigger_health_alert，直接给低烧处置建议
- 询问跌倒/骨折/血压/胸闷等处置方法 → 直接给出专业建议，结合参考知识（包括：是否可搬动、是否需要120等），禁止调用工具
- 用户只说"帮我记一下/记录"而未提设备控制或告警 → 只调 add_health_record，禁止同时调 read_environment_sensor

复合任务：一次性调用所有需要的工具，不分多轮。用户同时提到记录和查询时，必须同时调 add_health_record 和 query_health_records，不能只调其中一个。
如有参考知识上下文，务必在回复中直接引用原文关键词（如"物理降温""布洛芬""拨打120""不要随意搬动"等），不能只用自己的措辞概括。
纯知识咨询（不涉及操作）直接回答，无需调用工具。"""

TOOLS = [
    control_smart_device,
    add_health_record,
    query_health_records,
    trigger_health_alert,
    query_alert_log,
    set_medication_reminder,
    read_environment_sensor,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# 工具名关键词映射，用于 fallback 解析
# 意图检测正则
_RECORD_INTENT = re.compile(
    r"帮.{0,6}(记|记录)|记一下|记录(一下|这个|下来)"
    # 陈述句：成员 + 时态词/报告动词 + 症状（需有时态或"说"，防止匹配问句）
    r"|(?:奶奶|爷爷|爸爸|妈妈|孩子|老人|家人|他|她).{0,8}(?:今天|现在|刚才|最近|又|说).{0,20}(?:发烧|头痛|头晕|血压|血糖|心率|跌倒|摔倒|胸闷|腹痛|咳嗽|失眠|睡不着|睡眠|水肿|发热|疲惫)"
)
_ALERT_INTENT  = re.compile(r"赶紧.{0,4}(告|通知|联系)|晕倒|叫不醒|摔倒了.{0,4}(快|帮|赶)|快.{0,4}告警|立即告警|触发.{0,8}告警"
                             r"|并告警|帮.{0,8}告警|需要.{0,4}告警|发出.{0,4}告警")
_CONSULT_INTENT  = re.compile(r"怎么办|怎么处理|怎么判断|能.{0,4}吗|应该怎么|有什么建议|要告警吗|需要告警吗|吗[？?]")
_SENSOR_INTENT   = re.compile(r"(温度|湿度|空气质量|噪声|光照|PM2\.5).{0,6}(多少|怎么样|如何|是多少|读|看|查)"
                               r"|(现在|目前|当前).{0,6}(温度|湿度|空气|噪声|光照)")
_DELETE_INTENT  = re.compile(r"删(掉|除|了)|取消记录|撤销")

def _strip_rag(text: str) -> str:
    """移除RAG泄漏内容和未执行的JSON工具调用。"""
    text = re.sub(r"\[参考知识\].*", "", text, flags=re.DOTALL)
    # 清除模型将工具调用JSON原样输出为文本的情况
    text = re.sub(r'\{"name":\s*"\w+",\s*"arguments":\s*\{[^{}]*\}\}', "", text)
    return text.strip()
_DEVICE_IMPLICIT = re.compile(r"太暗|太亮|光线.{0,4}(暗|不足|差)|屋.{0,4}暗|太热|太冷|太吵")
_DEVICE_EXPLICIT = re.compile(
    r"(打开|开启|开一下|关闭|关掉|关上|关了|关一下|调亮|调暗|控制|锁上|解锁|开锁).{0,10}(灯|空调|窗帘|门锁|电视|风扇|门)"
    r"|把.{0,8}(灯|空调|温度|亮度|门锁|前门|大门|后门).{0,6}(开|关|调|设|锁)"
    r"|(卧室|客厅|厨房|书房).{0,6}(灯|空调).{0,4}(打开|关|开|调)"
    r"|调.{0,3}到\d+\s*度|亮度.{0,3}调.{0,3}\d+"
)
_QUERY_INTENT    = re.compile(r"查(一下|询|看|查).{0,6}(记录|状况|情况)|最近.{0,6}(记录|怎么样|状态)"
                               r"|有.{0,4}记录吗|之前.{0,8}记录|查.{0,2}(奶奶|爷爷|爸爸|妈妈|家人).{0,6}(记录|状况|历史)"
                               r"|他.{0,4}(有没有|之前|以前|历史).{0,4}记录|她.{0,4}(有没有|之前|以前|历史).{0,4}记录")
_ALERT_LOG_INTENT = re.compile(r"告警.{0,6}(日志|记录|历史|情况)|最近.{0,6}告警|查.{0,4}告警|告警.{0,4}查")
_REMINDER_INTENT = re.compile(
    # 明确设置意图："设置/添加/建一个 提醒/用药..."
    r"(设置|建立|加个|添加|建一个|帮.{0,4}设).{0,6}(提醒|用药|吃药|服药|药物提醒)"
    # 提醒+动作+药物词（窗口宽）
    r"|提醒.{0,20}(吃|服用?|用).{0,15}(药|片|粒|颗|胶囊|维生素|钙片)"
    # 每天/每日/时间段+动作+药物词
    r"|每.{0,6}(天|日|早|晚|餐|次).{0,20}(吃|服用?|用).{0,15}(药|片|粒|颗|胶囊|维生素)"
    # 记得/别忘/按时+吃药
    r"|(记得|别忘了?|不要忘|按时|定时).{0,15}(吃|服|用).{0,15}(药|片|粒|颗|维生素)"
    # 药物名+提醒/按时/每天等触发词
    r"|(降压药|降糖药|阿司匹林|维生素\w*|钙片|胰岛素|[^\s，。！？、]{2,4}药).{0,10}(提醒|记得|别忘|每天|每日|按时|定时)"
    # 快捷词
    r"|用药提醒|吃药提醒|服药提醒|服药计划|用药计划|用药安排"
    # 修改已有提醒
    r"|(降压药|降糖药|阿司匹林|[^\s，。]{2,4}药).{0,6}(改到|调整|换到|改为)"
)

_TOOL_KEYWORDS = {
    "control_smart_device": ["device_name", "action", "value"],
    "add_health_record": ["member_name", "symptom"],
    "query_health_records": ["member_name"],
    "trigger_health_alert": ["alert_type", "detail"],
    "set_medication_reminder": ["medication", "schedule"],
    "read_environment_sensor": ["room", "sensor_type"],
}


# 工具名→触发该工具所需的输入关键词（至少命中1个才保留 Fallback 调用）
_TOOL_TRIGGER_WORDS: dict[str, list[str]] = {
    "control_smart_device":  ["灯", "空调", "温度", "亮度", "窗帘", "门锁", "锁", "开", "关", "调", "设备"],
    "add_health_record":     ["记", "记录", "记一下", "症状", "头痛", "发烧", "血压", "血糖", "心率", "关节",
                              "睡眠", "睡不着", "失眠", "咳嗽", "头晕", "胸闷", "腹痛", "水肿", "疲惫"],
    "query_health_records":  ["查", "查询", "历史", "最近", "之前", "怎么样", "有没有"],
    "trigger_health_alert":  ["告警", "摔倒", "跌倒", "晕倒", "高烧", "昏迷", "叫不醒", "紧急"],
    "set_medication_reminder": ["提醒", "吃药", "服药", "用药", "药"],
    "read_environment_sensor": ["温度", "湿度", "空气", "噪声", "光照", "传感器", "PM2.5"],
}


_MEMBER_WORDS = ["爷爷", "奶奶", "爸爸", "妈妈", "孩子", "老爷子", "老人", "姥姥", "姥爷", "外公", "外婆",
                 "女朋友", "男朋友", "老伴", "老公", "老婆", "儿子", "女儿", "儿媳", "媳妇", "哥哥", "姐姐", "弟弟", "妹妹"]


def _extract_symptom(user_input: str) -> str:
    """从用户原话中提取症状描述，去除记录意图词和后续行动指令。"""
    # 去掉测量/观察动作前缀，如"刚给奶奶量了血压" "刚测了体温"
    s = re.sub(r"刚?(给.{0,4})?(量了|测了|查了|量过|测过|测的|量的)\s*", "", user_input)
    # 去掉记录意图词
    s = re.sub(
        r"帮.{0,6}(记|记录)(一下|这个\w*|这些?症状?|下来|下|了|症状)?|记一下|记录(一下|这个\w*|下来|症状)?|录这个\w*|录症状?|录下来|录一下|记下来",
        "", s
    )
    # 在行动分隔词或行动关键词处截断
    s = re.split(
        r"[，,]\s*(?:另外|再|同时|顺便|帮我|帮|然后|请|还要|最后|接着)"
        r"|[，,]?\s*(?:告警|触发(?:告警)?|通知家人|联系医生?|拨打120"
        r"|把.{0,8}(?:空调|灯|门锁|窗帘|风扇).{0,6}(?:调|开|关|设))"
        r"|[，,]?\s*(?:超高了|超低了|超标了|太高了|太低了|好高|好低|好严重|很严重|赶紧|快来|快叫|怎么得了)",
        s
    )[0]
    # 去掉末尾残留的"录/并/且"等动词碎片
    s = re.sub(r"[，,]?\s*录[并且]?$|[，,]?\s*(并|且|然后|同时)$", "", s)
    # 去掉孤立的"症状/健康状态"等填充词
    s = re.sub(r"[，,]\s*(症状|健康状态|身体状况)\s*(?=[，,]|$)", "", s)
    # 去掉尾部疑问/咨询语气词，如"怎么办""应该怎么"
    s = re.sub(r"[，,]?\s*(怎么办|应该怎么|该怎么|怎么处理)[？?！!。,，\s]*$", "", s)
    s = re.sub(r"[，。,.！!？?\s]+$", "", s).strip()
    # 去掉开头残留标点（如"：爷爷..." "，爷爷..."）
    s = re.sub(r"^[：:，,。.！!？?\s]+", "", s).strip()
    # 去掉成员称谓前的人称前缀，如"我奶奶"→"奶奶"
    s = re.sub(r"^[我他她]\s*(?=(" + "|".join(_MEMBER_WORDS) + r"))", "", s).strip()
    # 长输入前半段是闲聊时，从成员名称处截取（如T046场景）
    for mw in _MEMBER_WORDS:
        pos = s.find(mw)
        if pos > 15:   # 成员名在15字之后，前面是不相关内容
            s = s[pos:]
            break
    return s or user_input


def _force_extract(tool_name: str, user_input: str) -> list[dict]:
    """Channel3 兜底：直接从用户输入中提取参数，强制构造工具调用，不依赖 LLM。"""
    if tool_name == "add_health_record":
        member = next((w for w in _MEMBER_WORDS if w in user_input), "家庭成员")
        symptom = _extract_symptom(user_input)
        return [{"name": "add_health_record", "args": {"member_name": member, "symptom": symptom}, "id": "c3_0"}]
    if tool_name == "trigger_health_alert":
        member = next((w for w in _MEMBER_WORDS if w in user_input), "家庭成员")
        if re.search(r"摔|跌|倒", user_input):
            alert_type = "fall_detected"
        elif re.search(r"心跳|心率|心律|心悸|心脏", user_input):
            alert_type = "irregular_heartbeat"
        elif re.search(r"血压", user_input):
            alert_type = "abnormal_bp"
        elif re.search(r"发烧|高烧|体温|3[89][\.\d]*\s*度", user_input):
            alert_type = "high_fever"
        elif re.search(r"无活动|没有动|没动|长时间未动|无活动", user_input):
            alert_type = "no_activity"
        else:
            alert_type = "emergency"
        quoted = re.search(r"['‘’“”'\"]([\s\S]{5,80})['‘’“”'\"]", user_input)
        detail = quoted.group(1).strip() if quoted else _extract_symptom(user_input)[:60]
        calls = [{"name": "trigger_health_alert", "args": {"member_name": member, "alert_type": alert_type, "detail": detail}, "id": "c3_0"}]
        # 高烧/发烧同时记录健康状态
        if alert_type == "high_fever":
            calls.append({"name": "add_health_record", "args": {"member_name": member, "symptom": user_input[:50]}, "id": "c3_1"})
        return calls
    if tool_name == "read_environment_sensor":
        room_map = {"卧室": "bedroom", "客厅": "living_room", "厨房": "kitchen", "浴室": "bathroom", "书房": "study"}
        room = next((v for k, v in room_map.items() if k in user_input), "living_room")
        sensor_map = {"温度": "temperature", "湿度": "humidity", "空气": "air_quality", "噪声": "noise", "光照": "light"}
        sensor = next((v for k, v in sensor_map.items() if k in user_input), "temperature")
        return [{"name": "read_environment_sensor", "args": {"room": room, "sensor_type": sensor}, "id": "c3_0"}]
    if tool_name == "control_smart_device":
        # 只支持已接入的设备：卧室灯、客厅灯、卧室空调、客厅空调
        room_map = {"卧室": "bedroom", "客厅": "living_room"}
        room_key = next((k for k in room_map if k in user_input), "客厅")
        room_val = room_map[room_key]
        # 隐式环境描述（优先提取显式数值）
        pct_m = re.search(r"(\d+)\s*%", user_input)
        if re.search(r"太暗|光线.{0,4}(暗|不足|差)|屋.{0,4}暗", user_input):
            val = pct_m.group(1) if pct_m else "80"
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{room_val}_light", "action": "set_brightness", "value": val},
                     "id": "c3_0"}]
        if re.search(r"太亮", user_input):
            val = pct_m.group(1) if pct_m else "30"
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{room_val}_light", "action": "set_brightness", "value": val},
                     "id": "c3_0"}]
        if re.search(r"太热", user_input):
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{room_val}_ac", "action": "set_temperature", "value": "26"},
                     "id": "c3_0"}]
        if re.search(r"太冷", user_input):
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{room_val}_ac", "action": "set_temperature", "value": "28"},
                     "id": "c3_0"}]
        if re.search(r"太吵", user_input):
            return [{"name": "control_smart_device",
                     "args": {"device_name": "windows", "action": "close"},
                     "id": "c3_0"}]
        # 灯控：仅支持卧室/客厅，不含书房和亮度调节
        is_off = bool(re.search(r"关|关闭|关掉|关上|关了|关灯|关一下", user_input))
        is_on  = bool(re.search(r"打开|开启|开灯|开一下", user_input))
        if re.search(r"灯|灯光", user_input) and not re.search(r"空调|温度|风扇|窗帘", user_input):
            LIGHT_ROOMS = {"卧室": "bedroom", "客厅": "living_room"}
            if re.search(r"所有.{0,4}(灯|灯光)", user_input):
                action = "turn_off" if is_off else "turn_on"
                return [{"name": "control_smart_device",
                         "args": {"device_name": f"{r}_light", "action": action}, "id": f"c3_{i}"}
                        for i, r in enumerate(["bedroom", "living_room"])]
            mentioned_lights = [(k, v) for k, v in LIGHT_ROOMS.items() if k in user_input]
            if len(mentioned_lights) >= 2:
                action = "turn_off" if is_off else "turn_on"
                return [{"name": "control_smart_device",
                         "args": {"device_name": f"{v}_light", "action": action}, "id": f"c3_{i}"}
                        for i, (k, v) in enumerate(mentioned_lights)]
            light_room = next((v for k, v in LIGHT_ROOMS.items() if k in user_input), None)
            if light_room is None:
                return []  # 没指定房间 → 触发反问
            action = "turn_off" if is_off else "turn_on"
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{light_room}_light", "action": action}, "id": "c3_0"}]
        # 空调温度
        temp_m = re.search(r"(\d+)\s*度|调.{0,3}到\s*(\d+)", user_input)
        if temp_m:
            val = next(g for g in temp_m.groups() if g)
            return [{"name": "control_smart_device",
                     "args": {"device_name": f"{room_val}_ac", "action": "set_temperature", "value": val},
                     "id": "c3_0"}]
    if tool_name == "query_health_records":
        member = next((w for w in _MEMBER_WORDS if w in user_input), None)
        if not member:
            # 尝试提取非标准称谓（如"小孙子"、"老伴"等2-4字名词）
            m = re.search(r"(小\w{1,3}|老\w{1,2}|\w{1,2}(子|女|儿|婆|公|哥|姐|弟|妹))", user_input)
            member = m.group(1) if m else "家庭成员"
        n_m = re.search(r"(\d+)\s*条", user_input)
        args: dict = {"member_name": member}
        if n_m:
            args["recent_n"] = int(n_m.group(1))
        return [{"name": "query_health_records", "args": args, "id": "c3_0"}]
    if tool_name == "set_medication_reminder":
        member = next((w for w in _MEMBER_WORDS if w in user_input), "家庭成员")
        # 优先匹配具体药名（避免通用模式提前命中"后吃降糖药"等垃圾片段）
        _KNOWN_DRUGS = (r"降压药|降糖药|降脂药|阿司匹林|钙片|维生素[A-Za-z0-9]*"
                        r"|胰岛素|布洛芬|止痛药|消炎药|感冒药|安眠药|头孢\w*|青霉素"
                        r"|健胃消食片|藿香正气水?|板蓝根|感冒灵|氯雷他定|二甲双胍"
                        r"|辛伐他汀|阿托伐他汀|华法林|硝苯地平|硝酸甘油|氨氯地平")
        med_m = re.search(_KNOWN_DRUGS, user_input)
        if not med_m:
            # 通用兜底1：末字为"药"（如"降糖药"等通用药名）
            all_m = list(re.finditer(r"[^\s，。！？、]{1,3}[^用吃服买换停开取\s，。！？、]药", user_input))
            med_m = all_m[-1] if all_m else None
        medication = med_m.group(0) if med_m else None
        if not medication:
            # 通用兜底2：动词后跟以片/颗/胶囊/丸结尾的中文词（如"吃健胃消食片"）
            eat_m = re.search(r"(?:吃|服用?|用)\s*([一-龥]{2,8}[片颗胶囊丸剂])", user_input)
            medication = eat_m.group(1) if eat_m else "药物"
        times = []
        if re.search(r"早上|早晨|早", user_input): times.append("早上")
        if re.search(r"晚上|睡前|晚", user_input): times.append("晚上")
        if re.search(r"中午|午饭", user_input):    times.append("中午")
        if re.search(r"三餐|饭后|饭前", user_input): times = ["三餐"]
        # 提取具体时间点，如"8点"、"20:00"
        time_point = re.search(r"(\d{1,2})[:：点](\d{0,2})", user_input)
        cnt_m = re.search(r"(\d+)\s*次", user_input)
        if times and time_point:
            h = time_point.group(1)
            m2 = time_point.group(2) or "00"
            schedule = "、".join(dict.fromkeys(times)) + f"{h}:{m2.zfill(2)}"
        elif times:
            schedule = "、".join(dict.fromkeys(times))
        elif time_point:
            h = time_point.group(1)
            m2 = time_point.group(2) or "00"
            schedule = f"每天{h}:{m2.zfill(2)}"
        else:
            schedule = f"每天{cnt_m.group(1)}次" if cnt_m else "每天"
        # 关键信息缺失时拒绝存储，返回空触发反问
        if medication == "药物" or member == "家庭成员":
            return []
        return [{"name": "set_medication_reminder",
                 "args": {"member_name": member, "medication": medication, "schedule": schedule},
                 "id": "c3_0"}]
    return []


def _extract_tool_calls_from_text(text: str, user_input: str = "") -> list[dict]:
    """从模型的文本输出中提取工具调用（fallback 解析器）。
    user_input 用于过滤与当前输入无关的幻觉工具调用。
    """
    calls = []
    pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"(?:arguments|parameters)"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    for match in re.finditer(pattern, text, re.DOTALL):
        tool_name = match.group(1)
        if tool_name not in TOOLS_BY_NAME:
            continue
        # 如果有用户输入，验证该工具调用是否与输入相关
        if user_input:
            trigger_words = _TOOL_TRIGGER_WORDS.get(tool_name, [])
            if trigger_words and not any(w in user_input for w in trigger_words):
                continue  # 输入里没有该工具的任何触发词，跳过（幻觉调用）
        try:
            args = json.loads(match.group(2))
            if "value" in args and not isinstance(args["value"], str):
                args["value"] = str(args["value"])
            # 设备控制：过滤掉 device_name 不在用户输入中的幻觉调用
            if tool_name == "control_smart_device" and user_input:
                # 保留调用条件：同名工具不超过1条（只取第一条）
                if any(c["name"] == "control_smart_device" for c in calls):
                    continue
            calls.append({"name": tool_name, "args": args, "id": f"fallback_{len(calls)}"})
        except json.JSONDecodeError:
            pass
    return calls


class SmartHomeAgent:
    def __init__(self, fresh_context: bool = False):
        import os
        if os.environ.get("XINGHU_USE_NPU") == "1":
            from .llm_rknn import ChatRKNN
            llm = ChatRKNN()
            print("[Agent] 推理后端: RK3588S NPU (rkllm)")
            print("[Agent] 正在加载 NPU 模型，请稍候...")
            llm._get_rt()  # 预加载，避免首次对话卡顿
            print("[Agent] NPU 模型已就绪 ✓")
        else:
            kwargs = {"model": "qwen2.5:1.5b", "temperature": 0, "num_ctx": 2048}
            if fresh_context:
                kwargs["keep_alive"] = 0  # 测试模式：禁用 KV-cache，消除跨用例污染
            llm = ChatOllama(**kwargs)
        self.llm = llm.bind_tools(TOOLS)
        self.history: list = [SystemMessage(content=SYSTEM_PROMPT)]
        self.profiler = Profiler()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolMessage]:
        """执行工具调用并返回 ToolMessage 列表。"""
        messages = []
        for tc in tool_calls:
            args = dict(tc.get("args", {}))
            # 无论哪个 Channel 调用，都清洗症状字段，防止原始输入被存入档案
            if tc["name"] == "add_health_record" and "symptom" in args:
                args["symptom"] = _extract_symptom(args["symptom"])
            tool = TOOLS_BY_NAME[tc["name"]]
            result = tool.invoke(args)
            print(f"  ← 返回: {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return messages

    def _trim_history(self):
        """多轮对话超过2条消息后，压缩旧消息，防止 2048 token 窗口溢出。"""
        non_system = [m for m in self.history if not isinstance(m, SystemMessage)]
        if len(non_system) <= 2:
            return
        new_history = [self.history[0]]  # 保留 SystemMessage
        # 最新 2 条非系统消息保持完整，其余全部压缩
        keep_full = set(id(m) for m in non_system[-2:])
        for msg in self.history[1:]:
            if id(msg) in keep_full:
                new_history.append(msg)
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                from langchain_core.messages import AIMessage as AI
                new_history.append(AI(content=msg.content[:60] + "…" if len(msg.content) > 60 else msg.content))
            elif isinstance(msg, ToolMessage):
                new_history.append(ToolMessage(content=msg.content[:40] + "…" if len(msg.content) > 40 else msg.content,
                                               tool_call_id=msg.tool_call_id))
            else:
                new_history.append(msg)
        self.history = new_history

    def chat(self, user_input: str) -> str:
        # 多轮对话历史过长时压缩，防止 2048 token 窗口溢出
        self._trim_history()

        # RAG：检索本地知识库，注入上下文
        # 输入含明确操作指令时跳过 RAG，防止 token 增加导致复合工具遗漏
        _has_action = re.search(r"帮我.{0,10}(记录|调|设置|开|关|锁|告警|提醒)|记一下|调到\d+度|亮度.{0,4}\d+|调.{0,4}\d+度"
                                r"|太暗|太亮|太热|太冷|太吵|光线.{0,4}(暗|不足)", user_input)
        # 历史较长时缩减 RAG 注入量，为 KV-cache 留出余量
        non_sys_count = sum(1 for m in self.history if not isinstance(m, SystemMessage))
        rag_top_k = 1 if non_sys_count >= 4 else 2
        # 短追问（≤10字）且已有上下文时跳过RAG，避免注入无关知识覆盖对话语境
        _is_short_followup = len(user_input.strip()) <= 10 and non_sys_count > 0
        rag_context = "" if _is_short_followup else retrieve(user_input, top_k=rag_top_k)
        augmented_input = user_input
        if rag_context and not _has_action:
            augmented_input = f"{user_input}\n\n{rag_context}"

        # 删除意图拦截：直接返回，不进入工具调用循环
        if _DELETE_INTENT.search(user_input):
            self.history.append(HumanMessage(content=augmented_input))
            self.profiler.start()
            self.history.append(HumanMessage(content="[系统] 不支持删除操作。"))
            resp = "抱歉，我目前不支持删除健康记录。如需修正，可以重新记录最新状态。"
            self.history.append(AIMessage(content=resp))
            return resp

        self.history.append(HumanMessage(content=augmented_input))

        t0, m0 = self.profiler.start()
        tool_call_count = 0
        called_tools: set[tuple] = set()  # (tool_name, action) 防止重复调用
        success = False

        try:
            for iteration in range(5):
                print(f"\n[思考轮次 {iteration + 1}]")
                response: AIMessage = self.llm.invoke(self.history)
                self.history.append(response)

                # 优先使用结构化 tool_calls
                if response.tool_calls:
                    # 过滤完全重复的调用（同名同action），允许同一工具以不同参数调多次
                    def _call_key(tc):
                        return (tc["name"], tc.get("args", {}).get("action", ""))
                    new_calls = [tc for tc in response.tool_calls if _call_key(tc) not in called_tools]
                    if not new_calls:
                        print("[Agent 决策] 工具已全部调用过，终止循环。")
                        success = True
                        return response.content
                    print(f"[Agent 决策] 调用 {len(new_calls)} 个工具：")
                    for tc in new_calls:
                        print(f"  → {tc['name']}({tc['args']})")
                    tool_call_count += len(new_calls)
                    called_tools.update(_call_key(tc) for tc in new_calls)
                    tool_messages = self._execute_tool_calls(new_calls)
                    self.history.extend(tool_messages)
                    continue

                # fallback：解析模型文本中的工具调用 JSON（过滤已调用过的）
                _neg = re.search(r"不用|停药|停止|取消|撤销|不需要|不要.{0,4}提醒", user_input)
                fallback_calls = [
                    fc for fc in _extract_tool_calls_from_text(response.content, user_input)
                    if (fc["name"], fc.get("args", {}).get("action", "")) not in called_tools
                    and not (_neg and fc["name"] == "set_medication_reminder")
                ]
                if fallback_calls:
                    print(f"[Agent 决策][Fallback] 从文本解析到 {len(fallback_calls)} 个工具调用：")
                    for tc in fallback_calls:
                        print(f"  → {tc['name']}({tc['args']})")
                    tool_call_count += len(fallback_calls)
                    called_tools.update((tc["name"], tc.get("args", {}).get("action", "")) for tc in fallback_calls)
                    tool_messages = self._execute_tool_calls(fallback_calls)
                    self.history.extend(tool_messages)
                    # 让模型基于工具结果给出最终回复
                    final = self.llm.invoke(self.history)
                    self.history.append(final)
                    success = True
                    return final.content

                # Channel 3：意图兜底——检测漏调的必要工具，强制补调（复合任务最多补3轮）
                if iteration <= 3:
                    missed = None
                    _called_names = {name for name, _ in called_tools}
                    if (_RECORD_INTENT.search(user_input)
                            and "add_health_record" not in _called_names
                            and not _CONSULT_INTENT.search(user_input)):
                        missed = "add_health_record"
                    elif _ALERT_INTENT.search(user_input) and "trigger_health_alert" not in _called_names and not _CONSULT_INTENT.search(user_input):
                        missed = "trigger_health_alert"
                    elif _SENSOR_INTENT.search(user_input) and "read_environment_sensor" not in _called_names:
                        missed = "read_environment_sensor"
                    elif (_DEVICE_IMPLICIT.search(user_input) or _DEVICE_EXPLICIT.search(user_input)) and "control_smart_device" not in _called_names:
                        missed = "control_smart_device"
                    elif _QUERY_INTENT.search(user_input) and "query_health_records" not in _called_names:
                        missed = "query_health_records"
                    elif _ALERT_LOG_INTENT.search(user_input) and "query_alert_log" not in _called_names:
                        missed = "query_alert_log"
                    elif _REMINDER_INTENT.search(user_input) and "set_medication_reminder" not in _called_names \
                            and not re.search(r"不用|停药|停止|取消|撤销|不需要|不要.*提醒", user_input):
                        missed = "set_medication_reminder"
                    if missed:
                        print(f"[Agent 决策][Channel3] 检测到意图但未调工具({missed})，直接提取参数强制调用。")
                        self.history.pop()  # 移除错误的 AI 文字回复
                        # 对于"记录一下"等短小补记指令，从历史中找上文作为提取上下文
                        extract_input = user_input
                        if (missed == "add_health_record"
                                and len(user_input) <= 15
                                and not any(w in user_input for w in _MEMBER_WORDS)):
                            for msg in reversed(self.history[:-1]):
                                if isinstance(msg, HumanMessage) and len(msg.content) > 15:
                                    prev_text = msg.content.split("\n\n")[0]  # 去掉 RAG 注入部分
                                    extract_input = prev_text + " " + user_input
                                    break
                        forced_calls = _force_extract(missed, extract_input)
                        if forced_calls:
                            tool_call_count += len(forced_calls)
                            called_tools.update((tc["name"], tc.get("args", {}).get("action", "")) for tc in forced_calls)
                            tool_messages = self._execute_tool_calls(forced_calls)
                            self.history.extend(tool_messages)
                            continue  # 继续循环，允许后续迭代补调剩余工具
                        # 信息不足时主动反问，而不是存垃圾数据
                        if missed == "control_smart_device" and re.search(r"灯|灯光", user_input):
                            reply = "请问是卧室还是客厅的灯？"
                            self.history.append(AIMessage(content=reply))
                            success = True
                            return reply
                        if missed == "set_medication_reminder":
                            reply = "请问是给谁设置用药提醒？需要服用什么药物，什么时间服用？"
                            self.history.append(AIMessage(content=reply))
                            success = True
                            return reply
                        # 其他提取失败则回退到重推理
                        self.history.pop()  # 移除原始 HumanMessage
                        forced = f"{augmented_input}\n[必须调用 {missed} 工具]"
                        self.history.append(HumanMessage(content=forced))
                        continue

                # 无工具调用，直接回复
                print("[Agent 决策] 无需工具，直接回复。")
                success = True
                return _strip_rag(response.content)

            final = self.llm.invoke(self.history)
            self.history.append(final)
            success = True
            return _strip_rag(final.content)

        finally:
            latency, mem_delta = self.profiler.end(t0, m0, user_input, tool_call_count, success)
            print(f"\n[Perf] 耗时 {latency:.0f}ms | 内存增量 {mem_delta:+.1f}MB | 工具调用 {tool_call_count} 次")

    def reset(self):
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
