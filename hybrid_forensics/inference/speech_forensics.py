"""
Stage 1: SpeechForensics - Audio-Visual Synchronization Detection
Based on AV-HuBERT features
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import soundfile as sf
import os.path as osp
import tempfile
from argparse import Namespace

from .utils import extract_audio_features, normalize_audio_features, truncate_audio


class SpeechForensics:
    """
    SpeechForensics detector using AV-HuBERT
    
    Detects audio-visual desynchronization by comparing visual and audio features
    """
    
    def __init__(self, model, task_cfg, max_length=50):
        """
        Initialize SpeechForensics detector
        
        Parameters
        ----------
        model : torch.nn.Module
            AV-HuBERT model
        task_cfg : object
            Task configuration from checkpoint
        max_length : float
            Maximum video length in seconds
        """
        self.model = model
        self.task_cfg = task_cfg
        self.max_length = max_length
        self.tmp_dir = tempfile.mkdtemp()
    
    def extract_visual_feature(self, video_path):
        """
        Extract visual features from video using AV-HuBERT
        
        Parameters
        ----------
        video_path : str
            Path to video file
        
        Returns
        -------
        torch.Tensor
            Visual features
        """
        # Import av_hubert utilities
        import importlib.util
        avhubert_dir = osp.join(osp.dirname(__file__), '..', '..', 'av_hubert')
        avhubert_utils_path = osp.join(avhubert_dir, 'avhubert', 'utils.py')
        spec = importlib.util.spec_from_file_location("avhubert_utils", avhubert_utils_path)
        avhubert_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(avhubert_utils)
        
        # Load and preprocess frames
        transform = avhubert_utils.Compose([
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((self.task_cfg.image_crop_size, self.task_cfg.image_crop_size)),
            avhubert_utils.Normalize(self.task_cfg.image_mean, self.task_cfg.image_std)
        ])
        
        frames = avhubert_utils.load_video(video_path)
        
        # Get FPS and truncate if needed
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        
        if len(frames) > fps * self.max_length:
            frames = frames[:int(fps * self.max_length)]
        
        frames = transform(frames)
        frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        
        with torch.no_grad():
            feature, _ = self.model.extract_finetune(
                source={'video': frames, 'audio': None},
                padding_mask=None,
                output_layer=None
            )
            feature = feature.squeeze(dim=0)
        
        return feature
    
    def extract_audio_feature(self, audio_path):
        """
        Extract audio features from audio file using AV-HuBERT
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        
        Returns
        -------
        torch.Tensor
            Audio features
        """
        audio_feats = extract_audio_features(audio_path)
        audio_feats_tensor = normalize_audio_features(audio_feats)
        
        with torch.no_grad():
            feature, _ = self.model.extract_finetune(
                source={'video': None, 'audio': audio_feats_tensor},
                padding_mask=None,
                output_layer=None
            )
            feature = feature.squeeze(dim=0)
        
        return feature
    
    def calc_cos_dist(self, feat1, feat2, vshift=15):
        """
        Calculate cosine distance with temporal shift
        
        Parameters
        ----------
        feat1 : torch.Tensor
            Visual features
        feat2 : torch.Tensor
            Audio features
        vshift : int
            Temporal shift window
        
        Returns
        -------
        float
            Maximum cosine similarity
        """
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        if len(feat1) != len(feat2):
            sample = np.linspace(0, len(feat1)-1, len(feat2), dtype=int)
            feat1 = feat1[sample.tolist()]
        
        win_size = vshift * 2 + 1
        feat2p = F.pad(feat2, (0, 0, vshift, vshift))
        dists = []
        
        for i in range(len(feat1)):
            dist = F.cosine_similarity(
                feat1[[i], :].repeat(win_size, 1),
                feat2p[i:i+win_size, :]
            ).cpu().numpy()
            dists.append(dist)
        
        dists = np.asarray(dists)
        return float(dists.mean(axis=0).max())
    
    def detect(self, mouth_roi_path, audio_path, return_features=False):
        """
        Detect deepfake using audio-visual synchronization

        Parameters
        ----------
        mouth_roi_path : str
            Path to mouth ROI video
        audio_path : str
            Path to audio file
        return_features : bool
            If True, return intermediate features for CoDo-Net

        Returns
        -------
        float or tuple or None
            If return_features=False (default), returns similarity score.
            If return_features=True, returns (similarity, feature_dict) where
            feature_dict contains:
            - 'visual': [T, 1024] visual features
            - 'audio': [T, 1024] audio features
            - 'concat': [T, 2048] concatenated features for CoDo-Net
            Returns None on error.
        """
        try:
            # Truncate audio if needed
            audio_path = truncate_audio(audio_path, max_length=self.max_length)

            # Extract features
            visual_feature = self.extract_visual_feature(mouth_roi_path)
            audio_feature = self.extract_audio_feature(audio_path)

            visual_feature_cpu = visual_feature.detach().cpu()
            audio_feature_cpu = audio_feature.detach().cpu()

            # Calculate similarity
            similarity = self.calc_cos_dist(
                visual_feature_cpu,
                audio_feature_cpu
            )

            if return_features:
                # Align temporal dimensions if needed
                if len(visual_feature_cpu) != len(audio_feature_cpu):
                    sample = np.linspace(0, len(visual_feature_cpu)-1, len(audio_feature_cpu), dtype=int)
                    visual_aligned = visual_feature_cpu[sample.tolist()]
                    audio_aligned = audio_feature_cpu
                else:
                    visual_aligned = visual_feature_cpu
                    audio_aligned = audio_feature_cpu

                # Create concatenated features (key for CoDo-Net)
                concat_feature = torch.cat([visual_aligned, audio_aligned], dim=-1)

                feature_dict = {
                    'visual': visual_aligned,      # [T, 1024]
                    'audio': audio_aligned,        # [T, 1024]
                    'concat': concat_feature       # [T, 2048] - used by CoDo-Net
                }
                return similarity, feature_dict
            return similarity

        except Exception as e:
            print(f"  SpeechForensics error: {e}")
            return None
    
    def __del__(self):
        """Cleanup temporary directory"""
        import shutil
        if hasattr(self, 'tmp_dir') and osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
