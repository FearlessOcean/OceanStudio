# OceanStudio
## 项目经历

### 🏠 [SmartHome-Care-Agent](https://claude.ai/chat/SmartHome-Care-Agent)

**面向居家养老场景的端侧智能 Agent** | *中兴AI竞赛 区域优胜奖*

支持健康看护、异常告警与智能家居控制，部署于 RK3588S 开发板，NPU 加速推理。

- 完成设备控制、健康记录、用药提醒、环境传感器等 7 个工具模块开发
- 三通道工具调用机制（结构化解析 + 文本正则 + 参数强提取），工具调用成功率 **100%**
- NPU 推理较 CPU 模式**提速 18×**；平均响应 **557ms**；系统测试通过率 **97–100%**

```
Python` `LangChain` `Ollama` `Flask` `Streamlit` `RK3588S` `NPU
```

------

### 🤖 [RAG-CustomerService](https://claude.ai/chat/RAG-CustomerService)

**基于 LangChain 的 RAG 智能客服系统**

支持私有知识库接入的企业级智能问答系统，具备多轮对话与文档溯源能力。

- 混合部署架构：本地 Ollama（qwen3-embedding）向量化 + 云端 Gemini 2.5 Flash 生成
- MD5 哈希去重保证入库幂等性；元数据随向量一并存储，支持结果溯源
- `RunnableWithMessageHistory` 封装多轮对话链，session_id 隔离用户上下文，JSON 持久化

```
Python` `LangChain` `ChromaDB` `Ollama` `Gemini API` `Streamlit
```

------

### 💬 [Bert-Emotion-Classification](https://claude.ai/chat/Bert-Emotion-Classification)

**基于 BERT 的中文情感分类系统**

对 `bert-base-chinese` 进行端到端微调，数据集为电商评论（10类、万余条），完成正/负情感二分类，测试集准确率95%。

- 分层抽样划分训练/测试集，AutoTokenizer WordPiece 分词，95% 分位数截断序列长度
- BCEWithLogitsLoss + Adam 优化，TensorBoard 实时记录 Loss，按最优 Loss 自动保存 checkpoint
- 提供批量推理与交互式命令行预测两套接口

```
Python` `PyTorch` `HuggingFace Transformers` `TensorBoard
```

## Projects

### 🏠 [SmartHome-Care-Agent](https://claude.ai/chat/SmartHome-Care-Agent)

**End-side intelligent agent for elderly home care** | *ZTE AI Competition — Regional Award*

An on-device AI agent supporting health monitoring, anomaly alerts, and smart home control, deployed on RK3588S with NPU acceleration.

- 7 tool modules: device control, health records, medication reminders, environment sensors
- Triple-channel tool invocation (structured parsing + regex + forced extraction) — **100% call success rate**
- NPU deployment: **18× speedup** vs CPU; end-to-end response **avg 557ms**; test pass rate **97–100%**

```
Python` `LangChain` `Ollama` `Flask` `Streamlit` `RK3588S` `NPU
```

------

### 🤖 [RAG-CustomerService](https://claude.ai/chat/RAG-CustomerService)

**Enterprise-grade RAG customer service system with private knowledge base**

A full-stack intelligent Q&A system combining local embedding and cloud LLM generation, with multi-turn dialogue and document traceability.

- Hybrid architecture: local Ollama (qwen3-embedding) for vectorization + Gemini 2.5 Flash for generation
- MD5 deduplication for idempotent ingestion; metadata stored alongside vectors for result tracing
- Session-isolated multi-turn memory via `RunnableWithMessageHistory` + JSON persistence

```
Python` `LangChain` `ChromaDB` `Ollama` `Gemini API` `Streamlit
```

------

### 💬 [Bert-Emotion-Classification](https://claude.ai/chat/Bert-Emotion-Classification)

**Chinese sentiment classification via BERT fine-tuning**

End-to-end fine-tuning of `bert-base-chinese` on an e-commerce review dataset (10 categories, 10,000+ samples) for binary sentiment classification.

- Stratified train/test split with WordPiece tokenization and 95th-percentile sequence truncation
- BCEWithLogitsLoss + Adam; TensorBoard loss tracking; auto-checkpoint at best validation loss
- Dual inference interface: batch prediction + interactive CLI

```
Python` `PyTorch` `HuggingFace Transformers` `TensorBoard
```
