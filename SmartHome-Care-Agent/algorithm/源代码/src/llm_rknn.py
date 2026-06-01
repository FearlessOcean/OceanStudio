"""
llm_rknn.py —— RK3588S NPU 推理 LangChain 适配层
基于 librkllmrt.so ctypes 绑定，替换 ChatOllama，接口完全兼容。

使用前置条件（板子上）：
    sudo cp ~/rkllm/librkllmrt.so /usr/local/lib/
    sudo ldconfig
    XINGHU_USE_NPU=1 python algorithm/main.py
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# ── C 枚举常量 ──────────────────────────────────────────────────────────────────

RKLLM_RUN_NORMAL  = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH  = 2
RKLLM_RUN_ERROR   = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0


# ── C 结构体定义 ────────────────────────────────────────────────────────────────

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id",    ctypes.c_int32),
        ("embed_flash",       ctypes.c_int8),
        ("enabled_cpus_num",  ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch",           ctypes.c_uint8),
        ("use_cross_attn",    ctypes.c_int8),
        ("reserved",          ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path",        ctypes.c_char_p),
        ("max_context_len",   ctypes.c_int32),
        ("max_new_tokens",    ctypes.c_int32),
        ("top_k",             ctypes.c_int32),
        ("n_keep",            ctypes.c_int32),
        ("top_p",             ctypes.c_float),
        ("temperature",       ctypes.c_float),
        ("repeat_penalty",    ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty",  ctypes.c_float),
        ("mirostat",          ctypes.c_int32),
        ("mirostat_tau",      ctypes.c_float),
        ("mirostat_eta",      ctypes.c_float),
        ("skip_special_token",ctypes.c_bool),
        ("is_async",          ctypes.c_bool),
        ("img_start",         ctypes.c_char_p),
        ("img_end",           ctypes.c_char_p),
        ("img_content",       ctypes.c_char_p),
        ("extend_param",      RKLLMExtendParam),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role",            ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type",      ctypes.c_int),
        ("_u",              _InputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode",                ctypes.c_int),
        ("lora_params",         ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history",        ctypes.c_int),
    ]


class _LastHidden(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.c_void_p),
        ("embd_size",     ctypes.c_int),
        ("num_tokens",    ctypes.c_int),
    ]


class _Logits(ctypes.Structure):
    _fields_ = [
        ("logits",     ctypes.c_void_p),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class _PerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms",  ctypes.c_float),
        ("prefill_tokens",   ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens",  ctypes.c_int),
        ("memory_usage_mb",  ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text",              ctypes.c_char_p),
        ("token_id",          ctypes.c_int32),
        ("last_hidden_layer", _LastHidden),
        ("logits",            _Logits),
        ("perf",              _PerfStat),
    ]


# 回调函数类型
LLMResultCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)


# ── 底层运行时封装 ──────────────────────────────────────────────────────────────

class _RKLLMRuntime:
    """librkllmrt.so 的 Python 封装，单例使用。"""

    def __init__(self, lib_path: str = "/usr/local/lib/librkllmrt.so"):
        self._lib = ctypes.CDLL(lib_path)
        self._handle = ctypes.c_void_p(None)
        self._tokens: list[str] = []
        self._done   = threading.Event()
        self._lock   = threading.Lock()

        # 保持回调引用，防止 GC 回收
        self._cb_ref = LLMResultCallback(self._on_token)
        self._setup_sigs()

    def _setup_sigs(self):
        lib = self._lib
        lib.rkllm_createDefaultParam.restype  = RKLLMParam
        lib.rkllm_createDefaultParam.argtypes = []
        lib.rkllm_init.restype    = ctypes.c_int
        lib.rkllm_init.argtypes   = [ctypes.POINTER(ctypes.c_void_p),
                                      ctypes.POINTER(RKLLMParam),
                                      LLMResultCallback]
        lib.rkllm_run.restype     = ctypes.c_int
        lib.rkllm_run.argtypes    = [ctypes.c_void_p,
                                     ctypes.POINTER(RKLLMInput),
                                     ctypes.POINTER(RKLLMInferParam),
                                     ctypes.c_void_p]
        lib.rkllm_destroy.restype = ctypes.c_int
        lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        lib.rkllm_abort.restype   = ctypes.c_int
        lib.rkllm_abort.argtypes  = [ctypes.c_void_p]

    def _on_token(self, result_ptr, userdata, state):
        """每生成一个 token 被调用一次。"""
        if state in (RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING):
            if result_ptr and result_ptr.contents.text:
                token = result_ptr.contents.text.decode("utf-8", errors="replace")
                self._tokens.append(token)
        elif state in (RKLLM_RUN_FINISH, RKLLM_RUN_ERROR):
            self._done.set()
        return 0

    def load(self, model_path: str, max_context_len: int = 2048,
             max_new_tokens: int = 512, temperature: float = 0.0,
             top_k: int = 1, top_p: float = 0.9):
        param = self._lib.rkllm_createDefaultParam()
        param.model_path       = model_path.encode()
        param.max_context_len  = max_context_len
        param.max_new_tokens   = max_new_tokens
        param.temperature      = temperature
        param.top_k            = top_k
        param.top_p            = top_p
        param.skip_special_token = True
        param.is_async         = False

        handle = ctypes.c_void_p()
        ret = self._lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), self._cb_ref)
        if ret != 0:
            raise RuntimeError(f"rkllm_init 失败，错误码: {ret}")
        self._handle = handle
        print(f"[RKNN] 模型已加载: {model_path}")

    def run(self, prompt: str, timeout: float = 120.0) -> str:
        with self._lock:
            self._tokens.clear()
            self._done.clear()

            inp = RKLLMInput()
            inp.role            = b"user"
            inp.enable_thinking = False
            inp.input_type      = RKLLM_INPUT_PROMPT
            inp._u.prompt_input = prompt.encode("utf-8")

            infer = RKLLMInferParam()
            infer.mode         = RKLLM_INFER_GENERATE
            infer.lora_params  = None
            infer.prompt_cache_params = None
            infer.keep_history = 0  # 由 ChatML prompt 自带历史

            ret = self._lib.rkllm_run(
                self._handle,
                ctypes.byref(inp),
                ctypes.byref(infer),
                None,
            )
            if ret != 0:
                print(f"[RKNN] rkllm_run 错误 {ret}（可能上下文过长），返回空串让 Channel3 兜底")
                self._done.set()
                return ""

            self._done.wait(timeout=timeout)
            return "".join(self._tokens)

    def destroy(self):
        if self._handle and self._handle.value:
            self._lib.rkllm_destroy(self._handle)
            self._handle = ctypes.c_void_p(None)


# ── 消息格式转换（ChatML）──────────────────────────────────────────────────────

def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """把 LangChain message list 转成 qwen2.5 ChatML 格式字符串。"""
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif isinstance(msg, HumanMessage):
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            content = msg.content or ""
            if msg.tool_calls:
                import json
                for tc in msg.tool_calls:
                    content += (
                        f'\n{{"name": "{tc["name"]}", '
                        f'"arguments": {json.dumps(tc["args"], ensure_ascii=False)}}}'
                    )
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif isinstance(msg, ToolMessage):
            parts.append(f"<|im_start|>user\n[工具结果] {msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── LangChain ChatModel ────────────────────────────────────────────────────────

# 全局单例运行时（模型只加载一次）
_runtime: Optional[_RKLLMRuntime] = None
_runtime_lock = threading.Lock()


def _get_runtime(model_path: str, **kwargs) -> _RKLLMRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _RKLLMRuntime()
            _runtime.load(model_path, **kwargs)
    return _runtime


class ChatRKNN(BaseChatModel):
    """
    基于 RK3588S NPU 的 LangChain ChatModel。
    接口与 ChatOllama 完全兼容，agent.py 零改动切换。
    """

    model_path:     str   = os.environ.get(
        "XINGHU_MODEL_PATH",
        "/home/cat/models/qwen2.5-1.5b-w8a8-rk3588.rkllm",
    )
    max_context_len:int   = 2048
    max_new_tokens: int   = 512
    temperature:    float = 0.0
    top_k:          int   = 1
    top_p:          float = 0.9
    tool_injection: str   = ""  # bind_tools 后注入工具 Schema

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "rknn-llm"

    def _get_rt(self) -> _RKLLMRuntime:
        return _get_runtime(
            self.model_path,
            max_context_len=self.max_context_len,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        # 注入工具 Schema 到 SystemMessage
        if self.tool_injection:
            patched, injected = [], False
            for m in messages:
                if isinstance(m, SystemMessage) and not injected:
                    patched.append(SystemMessage(content=m.content + self.tool_injection))
                    injected = True
                else:
                    patched.append(m)
            messages = patched

        prompt = _messages_to_prompt(messages)
        rt     = self._get_rt()
        text   = rt.run(prompt)

        # 截断 stop token
        for s in (stop or ["<|im_end|>", "<|endoftext|>"]):
            if s in text:
                text = text[: text.index(s)]

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text.strip()))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        # 当前 rkllm_run 是同步的，流式直接走同步后返回
        result = self._generate(messages, stop, **kwargs)
        yield ChatGenerationChunk(
            message=AIMessage(content=result.generations[0].message.content)
        )

    def bind_tools(self, tools, **kwargs):
        """工具绑定：RKNN 模式不注入 Schema（依赖 Channel2/3 解析），避免超出上下文。"""
        return self

    @property
    def _identifying_params(self) -> dict:
        return {"model_path": self.model_path}
