# URIS：家居环境线上交互平台（研究导向）

> 面向家居场景的多模态交互模拟平台：摄像头输入 + 目标检测 + 场景记忆 + 指代消歧 + 多模态大模型问答/建议 + 结构化评估。
>
> 本项目当前定位是 **交互模拟与研究评估平台**，**不是机器人运动控制系统**（不做 ROS2 执行闭环）。
>
> 说明：旧版单文件“Video Reasoning Assistant”应用已保留在 `legacy/` 目录，当前根目录入口 `app.py` 指向新版平台。
> 历史根目录资料（旧版 Colab/单文件脚本、旧测试、旧指南）已归档到 `archive/legacy-video-reasoning-root/`；历史子项目已归档到 `archive/legacy-homerobot-core/`；大体积压缩包已归档到 `archive/model-artifacts/`；研究调研报告已整理到 `docs/research/`。

## 1. 项目定位（先说清楚）

这个项目的核心目标不是“驱动机器人去移动”，而是构建一个可研究、可演示、可评估的 **家居环境交互系统原型**：

- 平台直接接摄像头（实时预览 / 快照）
- 持续识别家居物体（YOLO 路线）
- 支持用户自然语言交互（问答 / 建议 / 澄清）
- 输出自然语言 + 结构化 JSON（用于评估与复现）
- 重点研究：**grounding、指代消歧、时间记忆、澄清策略、结构化输出可靠性、交互延迟**

这一定义非常重要，因为它决定了系统设计应优先选择“分层可解释架构”（detector + tracker + memory + VLM），而不是一上来做端到端运动控制。该取舍也与近期多模态系统研究和评估基准强调的问题高度一致（空间推理、幻觉、结构化输出、时序理解）[1][8][9][10][11][15]。

## 2. 功能复盘：哪些值得做，哪些当前不值得优先做

下面这部分是基于目前代码状态（截至 **2026-02-25**）的重新复盘。

### 2.1 值得做（研究核心，应该继续强化）

1. **YOLO +（可切换）Tracker + Object Registry 分层架构**
- 值得做，因为它把“识别错误”“跟踪不稳”“对话理解错误”拆开了，便于定位误差来源与做消融实验。
- 当前已实现 `simple` tracker，并预留 `ByteTrack / OC-SORT` 模式接口与降级机制（adapter-ready）。

2. **时间记忆（Temporal Memory）与事件摘要（Scene Events）**
- 值得做，因为家居交互不是单帧问题。用户经常依赖“刚才那个物体”“现在少了一个杯子”这类时间上下文。
- 当前已实现检测历史、事件日志和场景变化摘要（如 `count change` / `scene stable`）。

3. **指代消歧（Reference Resolution）+ 快速澄清策略**
- 值得做，而且是本项目最有研究价值的功能之一。
- 当前已支持如“那个杯子 / 左边那个杯子”的解析；在歧义场景下优先触发澄清而不是强行生成答案（提升可信性和交互体验）。

4. **结构化 JSON 输出 + Evaluation Lab 指标化评估**
- 值得做，因为这是项目从“demo”变成“研究平台”的关键一步。
- 当前已支持 `json_valid_rate`、`clarification_rate`、`reference_resolution_rate`、`cache_hit_rate` 等研究指标汇总。

5. **按需触发 Qwen（而非每帧跑 VLM）**
- 值得做，这是在线交互速度的关键。
- 当前实现采用：YOLO 高频、Qwen 低频按需触发（用户提问触发），并加入缓存和 Fast Mode。

### 2.2 值得做（工程支撑层，服务研究目标）

1. **模块化重构（`src/uris_platform`）**
- 值得做，因为便于后续接入不同 detector / tracker / VLM / prompt 版本做对比实验。

2. **性能开关（Fast Mode / Qwen Cache / Debug Prompt Bundle）**
- 值得做，因为它让“延迟-质量权衡”可以在 UI 中直接切换并被记录到评估数据。

3. **Live Camera 可视化状态栏 + Evaluation Lab**
- 值得做，因为它让论文中的指标和系统状态在演示阶段就可观察、可复现实验流程。

### 2.3 当前阶段不值得优先做（建议暂缓）

1. **机器人运动控制 / ROS2 执行闭环**
- 当前目标是交互模拟与研究评估，不是控制执行。
- 提前接入会显著增加系统复杂度，但不能直接提升你当前论文的核心贡献。

2. **每帧调用大模型进行实时理解**
- 在线体验会明显变差，且很难稳定评估。
- 分层架构下由 detector / tracker 先承担实时感知，大模型负责按需解释与建议，更适合你的项目目标。

3. **过早切到重型前后端分离（React + FastAPI + 服务网格）**
- 这类工程升级最终可能需要，但当前 Streamlit 平台已经足够支撑研究验证与 UI 演示。

## 3. 当前项目的研究点（建议作为主页重点）

本项目不是“再做一个多模态聊天 UI”，而是在研究一个更具体的问题：

### 研究问题（Research Questions）

- **RQ1：** 在家居场景在线交互中，`detector + tracker + object registry + VLM` 的分层系统，是否能在响应速度与可解释性上优于“直接用 VLM 看图回答”？
- **RQ2：** 时间记忆与对象注册表是否能显著提升指代消歧（如“那个杯子/左边那个”）的准确率与交互连贯性？
- **RQ3：** 当场景存在歧义时，优先“澄清”而不是“猜测回答”，是否能降低幻觉风险并提升用户体验？
- **RQ4：** 结构化 JSON 输出（尤其配合约束解码）是否能显著提升评估可复现性与系统可靠性？
- **RQ5：** 在可部署约束下（单机/工作站），如何取得识别率、交互质量与响应延迟之间的最优平衡？

这些问题与近期工作关注的方向高度一致：视觉定位与结构化输出能力（Qwen2.5-VL）[1][2][3]、多目标跟踪与身份稳定性（ByteTrack / OC-SORT）[6][7]、多模态幻觉诊断（HallusionBench）[8]、空间推理评估（EmbSpatial-Bench / SpatialEval）[9][10]、视频与时序理解评估（Video-MME）[11]。

## 4. 当前系统架构（代码已落地的主线）

```mermaid
flowchart LR
    A[Camera Input\nSnapshot / Optional WebRTC Preview] --> B[YOLO Detection]
    B --> C[Tracker Layer\nsimple / ByteTrack* / OC-SORT*]
    C --> D[Object Registry\nobj_id + track_id + bbox + status]
    D --> E[Temporal Memory\nDetection History + Scene Events]
    D --> F[Reference Resolution\n"那个杯子" / "左边那个"]
    E --> F
    F --> G[Qwen Adapter\nOn-demand only]
    B --> G
    G --> H[Natural Language Response]
    G --> I[Structured JSON]
    H --> J[Evaluation Lab]
    I --> J

    K[Perf Controls\nFast Mode / Cache / Debug] --> G

    classDef muted fill:#f5f5f5,stroke:#999,color:#333;
    class C muted;
```

说明：
- `*` 表示 **接口已预留、当前默认优雅降级到 `simple` tracker**（避免依赖缺失导致系统不可用）。
- Qwen 在 `Live Camera` 路径中采用按需触发与缓存策略，避免每帧大模型推理造成交互卡顿。

## 5. 当前已实现能力（按代码状态，不夸大）

### 5.1 平台能力（已实现）

- `Mission Control / Scenario Studio / Interaction Console / Live Camera / Operations / Evaluation Lab / Automation` 多标签页平台 UI
- `Live Camera`：浏览器摄像头快照检测（稳定路径） + 可选 WebRTC 预览
- YOLO 检测结果标准化与场景摘要
- Object Registry（`obj_id`、`track_id`、`bbox`、`center_norm`、`seen_count`、`mention_count`）
- Temporal Memory（检测历史、事件日志、场景变化摘要）
- 指代消歧（方向词 + 类别词 + 指示词）
- 歧义场景快速澄清（优先问清楚，而不是强行回答）
- Qwen Prompt（交互友好 + 学术化 + JSON 输出导向）
- Qwen 响应缓存（重复问题 + 同场景复用）
- Evaluation Lab（延迟、满意度、一致性、任务完成率、研究指标）

### 5.2 已实现但仍属“占位/降级优先”的能力（诚实说明）

- `ByteTrack / OC-SORT`：**模式入口与状态展示已接入**，当前默认降级到 `simple` tracker（真实后端依赖尚未接入）
- `Qwen Live Adapter`：已完成 prompt/解析/缓存/降级流程；在未满足本地模型依赖或运行条件时使用 fallback 响应路径

### 5.3 下一步最值得补强的能力（研究收益高）

1. **真实 ByteTrack/OC-SORT 后端接入**（提升 `object_id` 稳定性）[6][7]
2. **结构化解码（Outlines / XGrammar）接入**（提升 JSON 可靠性）[13][14][15]
3. **Hard split 评估集**（多同类、遮挡、低照、相似物体）
4. **置信度校准 / 拒答策略**（降低“自信地错”）[8]
5. **区域级理解增强**（检测 + 分割/裁剪联合解释）

## 6. 为什么这个系统设计有研究价值（而不只是工程堆功能）

### 6.1 它把“在线交互问题”拆成了可研究的子问题

相比端到端“把图像和问题都丢给大模型”，本系统把关键环节显式化：

- 感知（YOLO）
- 身份连续性（Tracker）
- 场景记忆（Object Registry + Temporal Memory）
- 指代解析（Reference Resolution）
- 生成与解释（Qwen）
- 结构化输出（JSON）
- 评估与日志（Evaluation Lab）

这种拆分的价值在于：
- 可以清晰做 ablation（例如去掉时间记忆/去掉澄清策略）
- 可以分层分析失败原因（漏检 vs 跟踪断轨 vs 指代错误 vs 语言幻觉）
- 更容易做“在线速度 vs 质量”的工程-研究折中

### 6.2 它直接对齐当前研究社区的真实痛点

- 多模态系统会“看错/想错/说得很自信”——这正是 HallusionBench 关心的问题 [8]
- 空间关系和具身相关表达（left/right/on 等）仍是 VLM 薄弱点——EmbSpatial-Bench / SpatialEval 都在强调这一点 [9][10]
- 时序视频理解和长上下文场景仍然困难——Video-MME 揭示了明显差距 [11]
- 实际部署中结构化输出的可靠性是硬需求——约束解码与 JSONSchemaBench 正在变成基础设施问题 [13][14][15]

换句话说：**你这个项目的研究点是“如何把现有模型与系统组件组合成一个可在线运行、可量化评估、可信度更高的家居交互平台”**，而不是单纯比较哪个模型参数更大。

## 7. 评估指标（当前平台已支持 / 建议继续扩展）

### 7.1 当前平台已支持（Evaluation Lab）

- 响应延迟：平均值 / 中位数 / p95
- 一致性：重复指令下推荐结果一致度
- 用户满意度（1–5）
- 任务完成率（仿真 + 人工覆盖）
- `json_valid_rate`
- `clarification_rate`
- `reference_resolution_rate`
- `cache_hit_rate`

### 7.2 建议补充（论文更有说服力）

- `tracking_stability`（ID switch proxy / IDF1 近似）
- `hallucination_flag`（人工标注或规则检测）
- `detection_stability`（短窗口类别波动）
- `reference_resolution_accuracy`（有 GT 的指代样本）
- `json_content_correctness`（不仅结构合法，还要内容正确）

## 8. 快速开始（当前主入口）

### 8.1 运行平台

```bash
make run
# 或
streamlit run app.py
```

### 8.2 推荐依赖（按需）

- 基础平台：`streamlit`, `pytest` 等（见 `requirements.txt`）
- 真实检测：`ultralytics`
- WebRTC 预览：`streamlit-webrtc`, `av`
- Qwen + LoRA 本地推理：`transformers`, `peft`（及可选量化/加速依赖）

### 8.3 当前已知运行策略（为了体验更稳）

- 优先使用 `Snapshot Camera (stable)` 路径做检测 + 交互
- YOLO 高频、Qwen 按需触发（用户提交问题时触发）
- 开启 `Fast Mode` 与 `Qwen Cache` 可显著改善体感延迟

## 9. 项目结构（研究相关部分）

```text
src/uris_platform/
├── streamlit_app.py                 # 主平台 UI（多标签页）
├── state.py                         # Session state（live memory / eval / perf）
├── prompts/
│   └── qwen_interaction_prompt.py   # 交互友好 + 学术化 prompt
├── services/
│   ├── vision_yolo.py               # YOLO 检测与标准化
│   ├── object_tracking.py           # Tracker 接口层（simple / bytetrack / ocsort）
│   ├── live_scene_memory.py         # Object registry + temporal memory + reference resolution
│   ├── qwen_adapter.py              # Qwen 适配器（缓存、解析、降级、澄清捷径）
│   └── evaluation.py                # Evaluation Lab 汇总指标
└── ui/
    ├── components.py                # UI 组件
    └── theme.py                     # 主题样式
```

## 10. 研究路线图（建议）

### Phase A（已完成基础）
- 平台化 UI、模块化架构、评估面板
- Live Camera + YOLO 检测 + Qwen 交互主流程
- Object Registry + Temporal Memory + 指代消歧

### Phase B（优先增强，最值得）
- 接入真实 ByteTrack / OC-SORT
- 结构化解码（Outlines / XGrammar）接入
- 构建 hard split 评估集与标注工具流

### Phase C（论文强化）
- 置信度校准 / 拒答策略实验
- Prompt / 模型版本 A/B 对比
- 离线大模型 teacher 做 error analysis（不影响在线体验）

### Phase D（可选扩展）
- 分割/区域级理解增强
- 语音输入（STT）
- 多摄像头 / 多房间评估

## 11. 参考文献与官方文档（论文引用）

> 以下引用用于说明本项目的研究背景、系统设计取舍和后续改进方向。优先使用论文原文、官方模型卡与官方文档。

1. **Qwen2.5-VL Technical Report** (arXiv, 2025)  
   https://arxiv.org/abs/2502.13923
2. **Qwen/Qwen2.5-VL-7B-Instruct** (Hugging Face Model Card)  
   https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
3. **Transformers: Qwen2.5-VL Docs** (Hugging Face)  
   https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
4. **Ultralytics YOLO Docs** (Official)  
   https://docs.ultralytics.com/
5. **Ultralytics Track Mode Docs** (Official)  
   https://docs.ultralytics.com/modes/track/
6. **ByteTrack: Multi-Object Tracking by Associating Every Detection Box** (ECCV 2022 / arXiv)  
   https://arxiv.org/abs/2110.06864  
   Code: https://github.com/ifzhang/ByteTrack
7. **Observation-Centric SORT (OC-SORT)** (CVPR 2023 / arXiv)  
   https://arxiv.org/abs/2203.14360  
   Code: https://github.com/noahcao/OC_SORT
8. **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models** (CVPR 2024 / arXiv)  
   https://arxiv.org/abs/2310.14566  
   Code: https://github.com/FuxiaoLiu/HallusionBench
9. **EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models** (ACL 2024)  
   https://aclanthology.org/2024.acl-short.33/
10. **SpatialEval: Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models** (NeurIPS 2024)  
    Project: https://spatialeval.github.io/  
    Code: https://github.com/jiayuww/SpatialEval
11. **Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis** (arXiv, 2024)  
    https://arxiv.org/abs/2405.21075
12. **TEACh: Task-driven Embodied Agents that Chat** (AAAI 2022 / arXiv)  
    https://arxiv.org/abs/2110.00534  
    Project: https://teachingalfred.github.io/
13. **Outlines (Structured Outputs / Constrained Generation)**  
    https://github.com/dottxt-ai/outlines
14. **XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models** (arXiv, 2024)  
    https://arxiv.org/abs/2411.15100  
    Code: https://github.com/mlc-ai/xgrammar
15. **Generating Structured Outputs from Language Models: Benchmark and Studies (JSONSchemaBench)** (arXiv, 2025)  
    https://arxiv.org/abs/2501.10868  
    Repo: https://github.com/guidance-ai/jsonschemabench

## 12. 致谢与说明

- 本项目当前使用/适配的核心方向是 `Qwen + YOLO + 场景记忆 + 评估面板`，重点在于 **系统设计与研究评估**，而非单模型炫技。
- 如果你希望做“论文版主页”，建议在本 README 基础上追加：数据集构建流程图、实验设置表、消融结果图、失败案例可视化。

---

如果你是研究者或开发者，欢迎基于本平台继续探索：
- 在线多模态交互的延迟-质量折中
- 指代消歧与场景记忆的系统设计
- 结构化输出可靠性与评估协议
