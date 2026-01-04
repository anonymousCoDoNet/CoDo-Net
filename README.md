# HybridForensics with CoDo-Net


```bash
#  preprocess
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

```

**Step 1: extract features**

```bash
python scripts/extract_features.py \
    --video_list data/train_list.txt \
    --semantic_mouth_dir data/cropped_mouths \
    --semantic_audio_dir data/audio \
    --dynamic_mouth_dir data/cropped_mouths \
    --output_dir data/cached_features/
```


**Step 2: train CoDo-Net**

```bash
# Phase 1: Warmup (3-5 epochs)

python -m hybrid_forensics.training.train_codo_net\     --mode warmup    --train_feature_dir /data/cached_features/train     --val_feature_dir /data/cached_features/val     --train_file_list /HybridForensics/train_list.txt     --val_file_list /HybridForensics/val_list.txt     --save_dir  ./checkpoints/codo_net     --epochs 10     --batch_size 32     --lr 1e-3


# Phase 2: Finetune (15-30 epochs)

python -m hybrid_forensics.training.train_codo_net     --mode finetune     --model_type codo     --projector_ckpt ./checkpoints/codo_net/codo_projector_warmup.pt     --train_feature_dir /data/codo_data/cached_features/train     --val_feature_dir /data/codo_data/cached_features/val     --train_file_list /train_list.txt     --val_file_list /val_list.txt     --save_dir /HybridForensics/checkpoints/codo_net     --epochs 20     --batch_size 16     --lr 1e-3     --projector_lr_scale 0.2     --weight_decay 1e-4     --seq_len 64     --num_workers 4
```

**Step 3: evaluate**

```bash
python scripts/evaluate_codo.py \
    --feature_dir data/cached_features/test \
    --file_list data/test_list.txt \
    --checkpoint checkpoints/codo_net/best_codo.pt \
    --output_dir results/codo
```



---

## Performance


| FakeAVCeleb | AVLips | IdForge | SemLip | 
|------|-----|----------|----------|
| **97.8%** | **97.9%** | **96.8%** | **97.5** | 


---



