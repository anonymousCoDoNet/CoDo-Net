"""
Main hybrid cascade detection pipeline
Combines SpeechForensics (Stage 1) and LipForensics (Stage 2)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import os.path as osp
from argparse import Namespace
from tqdm import tqdm
from sklearn import metrics

from .speech_forensics import SpeechForensics
from .lip_forensics import LipForensics
from ..models.model_loader import load_speech_model, load_lip_model
from ..config import resolve_path


def hybrid_cascade_detect(args):
    """
    Main hybrid cascade detection function
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing:
        - video_root: root directory of videos
        - file_list: path to file list
        - speech_mouth_dir: directory of speech mouth ROIs
        - speech_audio_dir: directory of speech audio
        - lip_mouth_dir: directory of lip mouth ROIs
        - output_dir: output directory for results
        - speech_threshold: threshold for SpeechForensics
        - lip_threshold: threshold for LipForensics
        - checkpoint_path: path to SpeechForensics model
        - lip_model_path: path to LipForensics model
        - max_length: maximum video length in seconds
    """
    
    # Resolve paths
    args.video_root = resolve_path(args.video_root)
    args.file_list = resolve_path(args.file_list)
    args.speech_mouth_dir = resolve_path(args.speech_mouth_dir)
    args.speech_audio_dir = resolve_path(args.speech_audio_dir)
    args.lip_mouth_dir = resolve_path(args.lip_mouth_dir)
    args.checkpoint_path = resolve_path(args.checkpoint_path)
    args.lip_model_path = resolve_path(args.lip_model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("HYBRID CASCADE DEEPFAKE DETECTION")
    print("="*60)
    print(f"Speech Threshold (SpeechForensics): {args.speech_threshold}")
    print(f"Lip Threshold (LipForensics): {args.lip_threshold}")
    print("="*60 + "\n")
    
    # Load models
    print("Loading models...")
    print("  Loading SpeechForensics (AV-HuBERT)...")
    speech_model, speech_task = load_speech_model(args.checkpoint_path)
    speech_detector = SpeechForensics(speech_model, speech_task.cfg, max_length=args.max_length)
    print("    SpeechForensics model loaded!")
    
    print("  Loading LipForensics (ResNet+MS-TCN)...")
    lip_model, lip_transform = load_lip_model(args.lip_model_path)
    lip_detector = LipForensics(lip_model, lip_transform)
    print("    LipForensics model loaded!")
    print("Models loaded successfully!\n")
    
    # Read video list
    with open(args.file_list, 'r') as f:
        video_list = f.read().split('\n')
    
    video_list = [v.strip() for v in video_list if v.strip()]
    print(f"Found {len(video_list)} videos to process\n")
    
    # Results storage
    results = {
        'stage1_results': [],
        'stage2_results': [],
        'skipped': []
    }
    
    all_scores = []
    all_predictions = []
    all_labels = []
    stage1_count = 0
    stage2_count = 0
    skipped_count = 0
    
    # Process videos
    for video_item in tqdm(video_list, desc="Processing videos"):
        if not video_item.strip():
            continue
        
        parts = video_item.split(' ')
        if len(parts) < 2:
            continue
        
        video_rel_path = parts[0]
        video_label = int(parts[1])
        
        # Construct paths
        # Note: Dual output structure from crop_mouths.py:
        #   output_dir/
        #   ├── video1/
        #   │   ├── speech_mouth.mp4    (for SpeechForensics)
        #   │   └── lip_mouth/          (for LipForensics)
        #   │       ├── 0000.png
        #   │       └── ...
        
        video_rel_path_no_ext = osp.splitext(video_rel_path)[0]
        
        # SpeechForensics: MP4 video file
        speech_mouth_path = osp.join(args.speech_mouth_dir, video_rel_path_no_ext, 'speech_mouth.mp4')
        
        # Audio file (same directory structure as video)
        speech_audio_path = osp.join(args.speech_audio_dir, video_rel_path_no_ext + '.wav')
        
        # LipForensics: PNG image directory
        lip_mouth_dir = osp.join(args.lip_mouth_dir, video_rel_path_no_ext, 'lip_mouth')
        
        # Check SpeechForensics data
        if not (osp.exists(speech_mouth_path) and osp.exists(speech_audio_path)):
            results['skipped'].append({
                'video': video_rel_path,
                'label': video_label,
                'reason': 'SpeechForensics data missing'
            })
            skipped_count += 1
            continue
        
        # ===== Stage 1: SpeechForensics =====
        similarity = speech_detector.detect(speech_mouth_path, speech_audio_path)
        
        if similarity is None:
            results['skipped'].append({
                'video': video_rel_path,
                'label': video_label,
                'reason': 'SpeechForensics processing error'
            })
            skipped_count += 1
            continue
        
        if similarity < args.speech_threshold:
            # Stage 1 decision: Fake
            prediction = 0
            stage = 1
            
            results['stage1_results'].append({
                'video': video_rel_path,
                'label': video_label,
                'prediction': prediction,
                'similarity': similarity,
                'stage': stage
            })
            
            # Use similarity directly as score for Real class
            score = similarity
            all_scores.append(score)
            all_predictions.append(prediction)
            all_labels.append(video_label)
            stage1_count += 1
        
        else:
            # ===== Stage 2: LipForensics =====
            # Check if PNG directory exists and contains PNG files
            if not osp.exists(lip_mouth_dir):
                results['skipped'].append({
                    'video': video_rel_path,
                    'label': video_label,
                    'reason': 'LipForensics PNG directory missing',
                    'similarity': similarity
                })
                skipped_count += 1
                continue
            
            # Check if PNG files exist
            try:
                png_files = [f for f in os.listdir(lip_mouth_dir) if f.endswith('.png')]
                if len(png_files) == 0:
                    results['skipped'].append({
                        'video': video_rel_path,
                        'label': video_label,
                        'reason': 'LipForensics PNG files missing',
                        'similarity': similarity
                    })
                    skipped_count += 1
                    continue
            except Exception as e:
                results['skipped'].append({
                    'video': video_rel_path,
                    'label': video_label,
                    'reason': f'LipForensics directory error: {str(e)}',
                    'similarity': similarity
                })
                skipped_count += 1
                continue
            
            logit = lip_detector.detect(lip_mouth_dir)
            
            if logit is None:
                results['skipped'].append({
                    'video': video_rel_path,
                    'label': video_label,
                    'reason': 'LipForensics processing error',
                    'similarity': similarity
                })
                skipped_count += 1
                continue
            
            # Stage 2 decision
            lip_prediction_internal = 1 if logit >= args.lip_threshold else 0
            prediction = 1 - lip_prediction_internal
            stage = 2
            
            results['stage2_results'].append({
                'video': video_rel_path,
                'label': video_label,
                'prediction': prediction,
                'similarity': similarity,
                'logit': logit,
                'stage': stage
            })
            
            # Invert logit to get Real class score
            score = -logit
            all_scores.append(score)
            all_predictions.append(prediction)
            all_labels.append(video_label)
            stage2_count += 1
    
    # Convert to arrays
    scores = np.asarray(all_scores)
    predictions = np.asarray(all_predictions)
    labels = np.asarray(all_labels)
    
    # Print statistics
    print("\n" + "="*60)
    print("STAGE-WISE STATISTICS")
    print("="*60)
    print(f"\nStage 1 (SpeechForensics): {stage1_count} videos")
    if stage1_count > 0:
        stage1_labels = [r['label'] for r in results['stage1_results']]
        stage1_preds = [r['prediction'] for r in results['stage1_results']]
        stage1_acc = np.mean(np.array(stage1_preds) == np.array(stage1_labels))
        print(f"  Accuracy: {stage1_acc:.4f} ({stage1_acc*100:.2f}%)")
    
    print(f"\nStage 2 (LipForensics): {stage2_count} videos")
    if stage2_count > 0:
        stage2_labels = [r['label'] for r in results['stage2_results']]
        stage2_preds = [r['prediction'] for r in results['stage2_results']]
        stage2_acc = np.mean(np.array(stage2_preds) == np.array(stage2_labels))
        print(f"  Accuracy: {stage2_acc:.4f} ({stage2_acc*100:.2f}%)")
    
    print(f"\nSkipped: {skipped_count} videos")
    
    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    
    total_processed = len(predictions)
    print(f"\nTotal processed: {total_processed}")
    print(f"  Real: {np.sum(labels == 1)}")
    print(f"  Fake: {np.sum(labels == 0)}")
    
    if total_processed > 0 and len(np.unique(labels)) > 1:
        # Calculate metrics
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        auc = metrics.auc(fpr, tpr)
        
        accuracy = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions, zero_division=0)
        recall = metrics.recall_score(labels, predictions, zero_division=0)
        f1 = metrics.f1_score(labels, predictions, zero_division=0)
        
        print(f"\n--- Continuous Score Metrics ---")
        print(f"AUC (video-level): {auc:.4f} ({auc*100:.2f}%)")
        
        print(f"\n--- Discrete Prediction Metrics ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        # Save results
        results_file = osp.join(args.output_dir, 'hybrid_cascade_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'speech_threshold': args.speech_threshold,
                    'lip_threshold': args.lip_threshold,
                    'note': 'AUC calculated using continuous scores (similarity and logit)'
                },
                'summary': {
                    'total_processed': total_processed,
                    'stage1_count': stage1_count,
                    'stage2_count': stage2_count,
                    'skipped': skipped_count
                },
                'metrics': {
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                },
                'stage1_results': results['stage1_results'],
                'stage2_results': results['stage2_results'],
                'skipped': results['skipped']
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nInsufficient data for metric calculation")


def detect_main(args):
    """
    Main entry point for detection
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    start_time = time.time()
    hybrid_cascade_detect(args)
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hybrid Cascade Deepfake Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--video_root',
        type=str,
        required=True,
        help='Root directory of videos'
    )
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help='Path to file list (format: "video_path label")'
    )
    parser.add_argument(
        '--speech_mouth_dir',
        type=str,
        required=True,
        help='Directory containing speech mouth ROIs'
    )
    parser.add_argument(
        '--speech_audio_dir',
        type=str,
        required=True,
        help='Directory containing speech audio'
    )
    parser.add_argument(
        '--lip_mouth_dir',
        type=str,
        required=True,
        help='Directory containing lip mouth ROIs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--speech_threshold',
        type=float,
        default=0.5,
        help='Threshold for SpeechForensics (similarity >= threshold -> Real)'
    )
    parser.add_argument(
        '--lip_threshold',
        type=float,
        default=-8.92,
        help='Threshold for LipForensics (logit >= threshold -> Fake)'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoints/av_hubert/large_vox_iter5.pt',
        help='Path to SpeechForensics model checkpoint (download from https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt)'
    )
    parser.add_argument(
        '--lip_model_path',
        type=str,
        default='checkpoints/lipforensics/lipforensics_ff.pth',
        help='Path to LipForensics model checkpoint'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=10,
        help='Maximum video length in seconds'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='lrs2',
        choices=['lrs2', 'lrw', 'fakeavcceleb', 'avlips'],
        help='Dataset type for LipForensics'
    )
    
    args = parser.parse_args()
    detect_main(args)
