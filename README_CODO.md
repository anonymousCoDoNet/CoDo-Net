# HybridForensics with CoDo-Net

**自适应融合 Deepfake 检测系统** 

---

##  项目简介

HybridForensics 是一个结合 **AV-HuBERT**（音画同步检测）和 **MS-TCN**（口型一致性检测）的混合检测系统。

**最新更新**: 我们实现了 **CoDo-Net (Consistency of Dynamics and Semantics Network)**，通过学习两个检测器的互补性，实现自适应融合。

---

##  核心特性

### 1. 三种检测模式

| 模式 | 描述 | AUC | 速度 | 适用场景 |
|------|------|-----|------|---------|
| **级联检测** | Stage 1 快速过滤 + Stage 2 精细分析 | 85-90% |  | 计算资源受限 |
| **Baseline 融合** | 简单特征拼接 | 78-85% |  | 快速原型 |
| **CoDo-Net**  | 自适应残差学习融合 | 92-96% |  | 高精度要求 |

### 2. CoDo-Net 创新点

-  **特征对齐**: 自动处理时间维度和通道维度不匹配
-  **残差学习**: 学习语义-动态一致性，而非简单拼接
-  **自适应门控**: 根据样本特点动态调整两个模型的权重
-  **时序注意力**: 定位视频中最可疑的时间段

---

##  安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- FFmpeg



##  快速开始

### 方式 1: 使用级联检测

```bash
# 1. 预处理
python -m hybrid_forensics.preprocessing.extract_landmarks \
    --video_root data/videos \
    --file_list data/file_list.txt \
    --output_dir data/landmarks

python -m hybrid_forensics.preprocessing.crop_mouths \
    --video_root data/videos \
    --landmarks_dir data/landmarks \
    --file_list data/file_list.txt \
    --output_dir data/cropped_mouths
    --mean_face data/20words_mean_face.npy

python -m hybrid_forensics.preprocessing.extract_audio \
    --video_root data/videos \
    --file_list data/file_list.txt \
    --output_dir data/audio

# 2. 运行检测
python -m hybrid_forensics.inference.detect \
    --video_root data/videos \
    --file_list data/file_list.txt \
    --speech_mouth_dir data/cropped_mouths \
    --speech_audio_dir data/audio \
    --lip_mouth_dir data/cropped_mouths \
    --output_dir results/
```

### 方式 2: 使用 CoDo-Net




这将自动执行：预处理 → 特征提取 → 训练 → 评估

#### 逐步运行

**Step 1: 提取特征（离线缓存）**

```bash
python scripts/extract_features.py \
    --video_list data/train_list.txt \
    --speech_mouth_dir data/cropped_mouths \
    --speech_audio_dir data/audio \
    --lip_mouth_dir data/cropped_mouths \
    --output_dir data/cached_features/
```

**Step 2: 训练 CoDo-Net**

```bash
# Phase 1: Warmup (3-5 epochs)

python -m hybrid_forensics.training.train_codo_net\     --mode warmup    --train_feature_dir data/codo_data/cached_features/train     --val_feature_dir /data/codo_data/cached_features/val     --train_file_list train_list.txt     --val_file_list val_list.txt     --save_dir HybridForensics/checkpoints/codo_net     --epochs 10     --batch_size 32     --lr 1e-3


# Phase 2: Finetune (15-30 epochs)

python -m hybrid_forensics.training.train_codo_net     --mode finetune     --model_type codo     --projector_ckpt HybridForensics/checkpoints/codo_net/codo_projector_warmup.pt     --train_feature_dir data/codo_data/cached_features/train     --val_feature_dir data/codo_data/cached_features/val     --train_file_list train_list.txt     --val_file_list val_list.txt     --save_dir checkpoints/codo_net     --epochs 20     --batch_size 16     --lr 1e-3     --projector_lr_scale 0.2     --weight_decay 1e-4     --seq_len 64     --num_workers 4
```

**Step 3: 评估**

```bash
python scripts/evaluate_codo.py \
    --feature_dir data/cached_features/test \
    --file_list test_list.txt \
    --checkpoint checkpoints/codo_net/best_codo.pt \
    --output_dir results/codo
```

>  **详细参数说明**: 

---

##  性能对比

在 FakeAVCeleb 数据集上的测试结果：

| 方法 | AUC | Accuracy | F1 Score | 推理速度 |


---

##  项目结构

```
HybridForensics/
├── hybrid_forensics/
│   ├── preprocessing/          # 预处理模块
│   │   ├── extract_landmarks.py
│   │   ├── crop_mouths.py
│   │   └── extract_audio.py
│   ├── inference/              # 推理模块
│   │   ├── detect.py           # 级联检测
│   │   ├── sementic_forensics.py # Stage 1
│   │   ├── dynamic_forensics.py    # Stage 2
│   │   ├── feature_alignment.py #  特征对齐
│   │   └── detect_codo.py      #  CoDo-Net 推理
│   ├── models/                 # 模型定义
│   │   ├── codo_net.py         #  CoDo-Net 架构
│   │   └── model_loader.py
│   ├── datasets/               # 数据集
│   │   └── feature_dataset.py  #  支持特征对齐
│   └── training/               # 训练脚本
│       └── train_codo_net.py   #  两阶段训练
├── checkpoints/                # 模型权重
│   ├── av_hubert/
│   ├── artifict/
│   └── codo_net/               #  CoDo-Net 权重
├── configs/
│   └── default_config.yaml
├── scripts/
│   ├── extract_features.py     #  离线特征提取
│   └── evaluate_codo.py        #  CoDo-Net 评估
├── docs/
└── README.md
```

---

##  CoDo-Net 架构

```
输入: Semantic Features [B, T, 2048] + Dynamic Features [B, T, 512]
  │
  ├─> Projector (2048→512)
  │     └─> Projected Semantic [B, T, 512]
  │
  ├─> Residual Learning
  │     └─> residual = (Dynamic - Projected)² [B, T, 512]
  │
  ├─> Adaptive Gating
  │     └─> gate = σ(Conv1D(residual, dynamic)) [B, T, 1]
  │
  ├─> Feature Fusion
  │     └─> fused = Dynamic + gate * residual [B, T, 512]
  │
  ├─> Temporal Attention
  │     └─> attention = softmax(Conv1D(fused)) [B, T, 1]
  │     └─> pooled = Σ(fused * attention) [B, 512]
  │
  └─> Classifier
        └─> logits [B, 2] (Real/Fake)
```

**核心思想**:
- **Projector**: 将语义特征（音画同步）投影到动态空间（口型特征）
- **Residual**: 捕获两个模态的不一致性（假视频的 residual 更大）
- **Gating**: 自适应决定何时更信任哪个模态
- **Attention**: 定位视频中最可疑的时间段

---

##  文档

| 文档 | 描述 |
|------|-|
| [CODO_SYSTEM_DESIGN.md](CODO_SYSTEM_DESIGN.md) |  系统设计文档 |

---

##  实验与可视化

### 消融实验

验证各模块的贡献：

```bash
# 移除 Gating（固定权重）
python -m hybrid_forensics.training.train_codo_net --model_type codo_no_gate ...

# 移除 Temporal Attention（使用 Mean Pooling）
python -m hybrid_forensics.training.train_codo_net --model_type codo_no_attention ...

# 使用 diff 替代 concat
python -m hybrid_forensics.training.train_codo_net --model_type codo_diff ...
```

### 可视化分析

```python
# 注意力热图
import matplotlib.pyplot as plt
attention = output.attention.squeeze().cpu().numpy()
plt.imshow(attention.reshape(1, -1), cmap='hot', aspect='auto')
plt.xlabel('Time Steps')
plt.title('Temporal Attention (Suspicious Moments)')
plt.colorbar()
plt.savefig('attention_heatmap.png')

# Gate 权重分布
gate_values = [output.gate.mean().item() for output in all_outputs]
plt.hist(gate_values, bins=50)
plt.xlabel('Gate Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Adaptive Gating Weights')
plt.savefig('gate_distribution.png')
```

