#!/usr/bin/env python3
"""
基于CSV数据集的landmark提取脚本
从successful_preprocessed_metadata.csv中读取视频信息并提取68个面部关键点
"""

import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp
import sys
from skimage import io
import face_alignment
from glob import glob
import json
import argparse
import pandas as pd
from pathlib import Path

def detect_save_landmark_68_csv(args):
    """从CSV文件中读取视频信息并提取landmark"""
    csv_path = args.csv_path
    video_root = args.video_root
    out_dir = args.out_dir
    
    # 读取CSV文件
    print(f"📖 读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📊 总样本数: {len(df)}")
    
    # 统计信息
    print(f"📊 标签分布: {df['label'].value_counts().to_dict()}")
    if 'category' in df.columns:
        print(f"📊 类别分布: {df['category'].value_counts().to_dict()}")
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    
    # 统计处理结果
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    print("🚀 开始提取landmark...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理视频"):
        # 获取视频路径
        if args.use_full_paths:
            video_path = row['original_path']
        else:
            video_path = osp.join(video_root, row['original_path'])
        
        # 生成输出路径
        video_id = row['video_id']
        segment_id = row['segment_id']
        out_path = osp.join(out_dir, f"{video_id}_{segment_id}.json")
        
        # 检查输出文件是否已存在
        if osp.exists(out_path) and not args.force:
            skipped_count += 1
            continue
        
        # 检查输入视频是否存在
        if not osp.exists(video_path):
            print(f"⚠️  视频文件不存在: {video_path}")
            failed_count += 1
            continue
        
        try:
            # 创建输出目录
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            
            # 读取视频帧
            frames = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  无法打开视频: {video_path}")
                failed_count += 1
                continue
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if len(frames) == 0:
                print(f"⚠️  视频为空: {video_path}")
                failed_count += 1
                continue
            
            frames = np.asarray(frames)
            
            # 提取landmark
            landmarks = {}
            for i in range(len(frames)):
                frame = frames[i]
                landmark = fa.get_landmarks(frame)
                if (landmark is not None) and (len(landmark) != 0):
                    landmark = landmark[0]
                    landmark = landmark.tolist()
                else:
                    landmark = None
                
                img_name = f'{i:04d}.jpg'
                landmarks[img_name] = landmark
            
            # 保存landmark
            with open(out_path, 'w') as f:
                json.dump(landmarks, f, indent=2)
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败 {video_path}: {e}")
            failed_count += 1
            continue
    
    # 输出统计结果
    print(f"\n📊 处理完成:")
    print(f"  ✅ 成功处理: {processed_count} 个视频")
    print(f"  ⏭️  跳过: {skipped_count} 个视频")
    print(f"  ❌ 失败: {failed_count} 个视频")
    print(f"  📁 输出目录: {out_dir}")

def create_file_list_from_csv(csv_path, output_file, video_root="", use_full_paths=False):
    """从CSV文件创建文件列表"""
    df = pd.read_csv(csv_path)
    
    with open(output_file, 'w') as f:
        for idx, row in df.iterrows():
            if use_full_paths:
                video_path = row['original_path']
            else:
                video_path = osp.join(video_root, row['original_path'])
            
            # 格式: video_path label
            f.write(f"{video_path} {row['label']}\n")
    
    print(f"📝 文件列表已保存到: {output_file}")
    print(f"📊 包含 {len(df)} 个视频")

def main():
    parser = argparse.ArgumentParser(
        description='基于CSV数据集的landmark提取脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主要参数
    parser.add_argument('--csv_path', type=str, required=True,
                       help='CSV数据集文件路径')
    parser.add_argument('--video_root', type=str, default='',
                       help='视频根目录 (如果不使用完整路径)')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='landmark输出目录')
    parser.add_argument('--use_full_paths', action='store_true',
                       help='使用CSV中的完整路径而不是相对路径')
    
    # 模型参数
    parser.add_argument('--face_detector', type=str, 
                       default='checkpoints/Resnet50_Final.pth',
                       help='人脸检测器路径')
    parser.add_argument('--face_predictor', type=str,
                       default='checkpoints/2DFAN4-cd938726ad.zip',
                       help='landmark预测器路径')
    
    # 其他参数
    parser.add_argument('--force', action='store_true',
                       help='强制重新处理已存在的文件')
    parser.add_argument('--create_file_list', type=str, default='',
                       help='创建文件列表并保存到指定路径')
    parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg',
                       help='ffmpeg路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not osp.exists(args.csv_path):
        print(f"❌ CSV文件不存在: {args.csv_path}")
        return
    
    # 检查模型文件
    if not osp.exists(args.face_detector):
        print(f"❌ 人脸检测器不存在: {args.face_detector}")
        print("请下载RetinaFace模型到checkpoints/目录")
        return
    
    # 创建文件列表 (如果请求)
    if args.create_file_list:
        create_file_list_from_csv(
            args.csv_path, 
            args.create_file_list, 
            args.video_root, 
            args.use_full_paths
        )
        return
    
    # 初始化face alignment
    print("🔧 初始化Face Alignment...")
    try:
        global fa
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector='retinaface',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            face_detector_kwargs={'path_to_detector': args.face_detector}
        )
        print("✅ Face Alignment初始化成功")
    except Exception as e:
        print(f"❌ Face Alignment初始化失败: {e}")
        return
    
    # 开始处理
    detect_save_landmark_68_csv(args)

if __name__ == '__main__':
    main()
