# HybridForensics with CoDo-Net

## Set-up AV-Hubert
```bash
# clone/install AV-Hubert
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../avhubert
# install additional files for AV-Hubert
mkdir -p content/data/misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O content/data/misc/20words_mean_face.npy
cd ../../

cp -r modification/retinaface preprocessing/face-alignment/face_alignment/detection
cp modification/landmark_extract.py preprocessing/face-alignment

# download avhubert checkpoint
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
mv self_large_vox_433h.pt av_hubert/avhubert/self_large_vox_433h.pt
```

## Installation
Main prerequisites:

* `Python 3.10.14`
* `pytorch=2.2.0` (older version for compability with AVHubert)
* `pytorch-cuda=12.4`
* `lightning=2.4.0`
* `torchvision>=0.17`
* `scikit-learn>=1.3.2`
* `pandas>=2.1.1`
* `numpy>=1.26.4`
* `pillow>=10.0.1`
* `librosa>=0.9.1`
* `dlib>=19.24.9`
* `skvideo>=1.1.10`
* `ffmpeg>=4.3`
* `certifi==2021.5.30`
* `joblib==1.0.1`
* `python-dateutil==2.8.2`
* `pytz==2021.1`
* `scipy==1.7.1`
* `six==1.16.0`
* `threadpoolctl==2.2.0`
* `tqdm==4.62.0`
* `typing-extensions==3.10.0.0`


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



