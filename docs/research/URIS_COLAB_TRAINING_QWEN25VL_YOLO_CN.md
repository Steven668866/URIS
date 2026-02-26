# URIS Colab 训练方案（按研究路线选择模型与数据集）

> 面向当前 URIS 平台（家居环境交互模拟，不做机器人运动控制）的训练方案。
>
> 目标：提升 `识别率（YOLO）`、`指代消歧/澄清策略/结构化输出（Qwen VLM）`，并与现有 `object registry + temporal memory + Evaluation Lab` 对齐。

## 1. 先说结论：训练什么模型？用什么数据集？

### 1.1 模型选择（推荐两阶段）

1. **感知层（先做）**：`YOLOv8s`（速度/精度平衡）
- 用于提升家居物体检测稳定性（直接影响 object registry、指代消歧）
- 如果显存紧张或追求更高 FPS，可先用 `YOLOv8n`

2. **理解层（主线）**：`Qwen2.5-VL-7B-Instruct` + `QLoRA/LoRA`
- 训练重点不是“识别物体类别”，而是：
  - 指代消歧（那个杯子 / 左边那个）
  - 澄清策略（不确定时先问）
  - 结构化 JSON 输出（`user_response + analysis_json`）
  - 基于检测结果与场景记忆的 grounded 回复

### 1.2 数据集选择（按你的研究路线）

#### A. YOLO（检测）训练数据：用于提升识别率

- **自采家居数据（必选，优先级最高）**
  - 覆盖：低照、遮挡、反光、多同类（两个杯子）、远近景、不同角度
- **COCO（家居相关类别子集）**
  - 用于泛化打底（如 `cup/chair/table/bottle/book` 等）
- **SUN RGB-D（建议）**
  - 室内场景分布更贴近你项目
- **可选：ScanNet 抽帧**
  - 强化室内布局变化与时序连续场景的检测鲁棒性

#### B. Qwen2.5-VL（交互/结构化输出）训练数据：用于提升研究指标

- **自采 URIS 家居交互数据（必选，核心）**
  - 包含：`image + user_query + object_registry + reference_resolution + target(user_response + analysis_json)`
- **RefCOCO / RefCOCO+ / RefCOCOg（转成 URIS 格式后使用）**
  - 用于学习指代表达与目标选择（注意标注噪声，建议清洗/抽样人工复核）
- **TEACh 对话切片（转成 URIS 格式后使用）**
  - 用于学习澄清/确认/纠错对话行为（不是运动控制）
- **JSON 对齐合成数据（建议）**
  - 专门训练 `analysis_json` 字段完整性、置信度与局限性表达

#### C. 这些数据更适合“评估”，不建议直接混入主 SFT 训练

- `EmbSpatial-Bench`, `SpatialEval`（空间推理评测）
- `Video-MME`（时序/视频理解评测）

## 2. 数据集配比建议（VLM）

建议从这个比例起步（可按你的样本规模调整）：

- `50%` 自采 URIS 家居交互数据（核心）
- `20%` RefCOCO 系列（已转 URIS 格式）
- `15%` TEACh 对话切片（已转 URIS 格式）
- `15%` JSON 对齐合成数据

说明：
- 你的研究目标是在线交互与评估，不是通用视觉描述，所以 **自采数据占比必须最高**。
- 公共数据集主要用于补“表达形态”和“多样性”，不是替代真实场景数据。

## 3. Colab 训练前准备（推荐环境）

### 3.1 推荐 Colab 环境

- **Qwen2.5-VL-7B + QLoRA**：建议 `Colab Pro/Pro+ A100`
- **YOLOv8s**：`T4 / L4 / A100` 都可

### 3.2 仓库内你要用到的文件（已准备好）

- 数据准备脚本：`/Users/shihaochen/github/URIS/scripts/colab/prepare_uris_vlm_dataset.py`
- YOLO 训练脚本：`/Users/shihaochen/github/URIS/scripts/colab/train_yolo_home_objects_colab.py`
- YOLO 数据集 YAML 模板：`/Users/shihaochen/github/URIS/configs/training/uris_home_objects_yolo.example.yaml`
- Qwen LoRA YAML 模板：`/Users/shihaochen/github/URIS/configs/training/qwen25vl_lora_colab.example.yaml`

---

## 4. Colab 代码（Qwen2.5-VL-7B-Instruct + QLoRA，主线）

> 这部分是你当前平台最重要的训练代码（提升 `指代消歧 / 澄清策略 / JSON 输出 / grounded interaction`）。
>
> 训练框架采用 `LLaMA-Factory`（方便做 VL LoRA/QLoRA）。

### Cell 1：挂载 Google Drive 并准备环境

```python
from google.colab import drive

drive.mount('/content/drive')
```

```bash
# 克隆你的项目（如果你已经上传了 zip，也可以解压）
cd /content
rm -rf URIS
git clone <你的仓库地址> URIS
cd URIS

# 安装训练依赖（建议单独升级）
pip install -U pip
pip install -U "torch" "torchvision" "torchaudio"
pip install -U "transformers>=4.49" "datasets" "accelerate" "peft" "bitsandbytes" "pillow"
pip install -U "trl"

# LLaMA-Factory（建议使用较新版本；如果版本冲突可改为固定 commit）
pip install -U "llamafactory"
```

### Cell 2：准备你的多源训练数据（按研究路线）

> 假设你已经把这些源数据转换成 URIS 中间格式（JSON / JSONL），每条至少包含：
> `image`, `user_query`, `target.user_response`, `target.analysis_json`。

```bash
cd /content/URIS

python scripts/colab/prepare_uris_vlm_dataset.py \
  --source self_home=/content/drive/MyDrive/URIS_DATA/vlm/self_home_interaction.jsonl \
  --source refcoco=/content/drive/MyDrive/URIS_DATA/vlm/refcoco_uris_transformed.jsonl \
  --source teach=/content/drive/MyDrive/URIS_DATA/vlm/teach_dialog_uris_transformed.jsonl \
  --source json_align=/content/drive/MyDrive/URIS_DATA/vlm/json_alignment_synth.jsonl \
  --images-root /content/drive/MyDrive/URIS_DATA/images \
  --out-dir /content/drive/MyDrive/URIS_DATA/llamafactory_qwen25vl_v2 \
  --dataset-name-prefix uris_vlm \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --split-key-field split_key \
  --language zh
```

执行后会生成（关键文件）：
- `train_sharegpt.json`
- `val_sharegpt.json`
- `test_sharegpt.json`
- `dataset_info.json`
- `prepare_summary.json`

### Cell 3：创建训练配置（Qwen2.5-VL + QLoRA）

> 注意：`template` 在不同版本 LLaMA-Factory 里可能是 `qwen2_vl`（常见）或变体名。若报错请查看版本文档/报错提示调整。

```bash
cat > /content/train_qwen25vl_uris_lora.yaml <<'YAML'
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
trust_remote_code: true
image_max_pixels: 262144
video_max_pixels: 16384

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
quantization_bit: 4

### dataset
dataset_dir: /content/drive/MyDrive/URIS_DATA/llamafactory_qwen25vl_v2
dataset: uris_vlm_train
template: qwen2_vl
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: /content/drive/MyDrive/URIS_CHECKPOINTS/Qwen2.5-VL-URIS-LoRA-v2
logging_steps: 10
save_steps: 200
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
dataloader_num_workers: 2

### optional eval (if your LLaMA-Factory version supports eval_dataset)
# do_eval: true
# eval_dataset: uris_vlm_val
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 200
YAML
```

### Cell 4：开始训练

```bash
cd /content/URIS
llamafactory-cli train /content/train_qwen25vl_uris_lora.yaml
```

### Cell 5：训练后快速检查（文件是否产出）

```bash
ls -lah /content/drive/MyDrive/URIS_CHECKPOINTS/Qwen2.5-VL-URIS-LoRA-v2
```

你希望至少看到：
- `adapter_config.json`
- `adapter_model.safetensors`

### Cell 6（可选）：导出/合并（按需要）

> 建议先保留 LoRA adapter，不急着 merge；这样后续做版本对比更方便。

```bash
# 可选：如果你需要导出合并模型，再根据当前 LLaMA-Factory 版本文档使用 export 命令
# llamafactory-cli export ...
```

---

## 5. Colab 代码（YOLOv8s，提升识别率）

> 这部分直接提升 `Live Camera` 的检测稳定性，从而间接提升 object registry 和指代消歧效果。

### Cell 1：安装依赖并准备数据集 YAML

```bash
cd /content/URIS
pip install -U ultralytics
```

将 `configs/training/uris_home_objects_yolo.example.yaml` 复制到你的 Drive 并按实际类别/路径修改，例如：

- `/content/drive/MyDrive/URIS_DATA/yolo_home_objects/uris_home_objects.yaml`

### Cell 2：训练 YOLO

```bash
cd /content/URIS
python scripts/colab/train_yolo_home_objects_colab.py \
  --data /content/drive/MyDrive/URIS_DATA/yolo_home_objects/uris_home_objects.yaml \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --project /content/drive/MyDrive/URIS_CHECKPOINTS/yolo_runs \
  --name home_objects_v1 \
  --export onnx
```

### Cell 3：产物检查

```bash
ls -lah /content/drive/MyDrive/URIS_CHECKPOINTS/yolo_runs/home_objects_v1/weights
```

重点文件：
- `best.pt`
- `last.pt`
- （可选）导出后的 `onnx` 文件

---

## 6. 训练顺序建议（与你的平台最匹配）

1. **先训练 YOLO（识别率和稳定性）**
- 先把感知层做好，`object registry` 和 `reference_resolution` 会更稳

2. **再训练 Qwen2.5-VL LoRA（交互策略和结构化输出）**
- 重点提升：`clarification_rate`, `reference_resolution_rate`, `json_valid_rate`

3. **最后在 URIS 平台里做 A/B 测试**
- 对比：旧 LoRA vs 新 LoRA
- 对比：旧 YOLO vs 新 YOLO
- 在 `Evaluation Lab` 记录指标变化

## 7. 关键注意事项（避免踩坑）

1. **Qwen VLM 微调不要指望替代 YOLO 检测**
- VLM 负责解释/澄清/结构化输出；实时检测交给 YOLO

2. **先统一标签体系**
- YOLO 类别名与 URIS alias（`cup/mug/杯子`）要对齐

3. **RefCOCO 系列有标注噪声**
- 建议抽样人工复核或只作为辅助数据，不要让它主导你的训练分布

4. **自采数据最重要**
- 你是家居摄像头场景，公开视频/图像数据只能做补充

5. **先把 JSON schema 训练稳定，再追求更复杂语言风格**
- 先提升结构可靠性与澄清策略，对论文和系统都更有价值

## 8. 你下一步可以直接做什么（最实用）

- 先用你已有 LoRA 数据格式，整理出一版 `self_home_interaction.jsonl`
- 用 `scripts/colab/prepare_uris_vlm_dataset.py` 跑通数据准备
- 在 Colab A100 跑 `Qwen2.5-VL-7B-Instruct + QLoRA` 训练 1 个 epoch 做 sanity check
- 再跑 YOLOv8s 的家居数据训练，优先看 `cup/chair/table` 三类提升情况

---

如果你要，我下一步可以继续给你：
- 一份 **`self_home_interaction.jsonl` 的标准样例模板**（完全对齐你平台 `analysis_json`）
- 一份 **RefCOCO / TEACh -> URIS 格式转换脚本**（批量转换用）
